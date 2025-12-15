from __future__ import annotations
import math
from typing import Optional,Tuple,Literal
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import argparse
import random

# 嘗試從當前目錄匯入模組，若失敗則從當前路徑匯入
try:
    from .mha import MultiHeadAttention
    from .position_encoding import get_positional_encoding
except Exception:
    from mha import MultiHeadAttention
    from position_encoding import get_positional_encoding

# 設定隨機種子，確保結果可重現
def set_seed(seed: int=1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 建立一個長度為 L 的序列資料集，輸入與輸出序列的開頭與結尾相同
def make_dataset_ends_equal(n: int, L: int, vocab: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """建立一個長度為L的序列資料集，輸入與輸出序列的開頭與結尾相同。"""
    X = torch.randint(1, vocab, (n, L))  # 隨機生成序列
    y= torch.zeros(n, dtype=torch.long)  # 初始化標籤
    half = n//2

    # 前一半數據的開頭與結尾相同
    X[:half,-1] = X[:half, 0]
    y[:half] = 1

    # 後一半數據的開頭與結尾不同
    for i in range(half, n):
        a = X[i,0].item()
        b = random.randint(1, vocab-2)
        if b>= a:
            b+=1
        X[i,-1] = b
        y[i] = 0
    idx = torch.randperm(n)
    # 打亂數據順序
    return X[idx], y[idx]

# 建立一個長度為 L 的序列資料集，輸入與輸出序列的第 i 與 j 個位置相同
def make_dataset_compare_ij(n:int, L:int, vocab: int,i: int, j:int) -> Tuple[torch.Tensor, torch.Tensor]:
    """建立一個長度為L的序列資料集，輸入與輸出序列的第i與j個位置相同。"""
    assert 0 <= i < L and 0 <= j < L and i != j
    X = torch.randint(0, vocab, (n, L))  # 隨機生成序列
    y = (X[:,i] > X[:,j]).long()  # 比較第 i 和第 j 個位置的值
    return X, y

# TinyConfig 類別，用於儲存模型的超參數
@dataclass
class TinyConfig:
    vocab: int = 100  # 詞彙表大小
    d_model: int = 128  # 嵌入維度
    num_heads: int = 4  # 注意力頭數
    dropout: float = 0.0  # dropout 機率

# TinyEncoder 類別，包含嵌入層、多頭注意力層、LayerNorm 和分類層
class TinyEncoder(nn.Module):
    def __init__(self,cfg:TinyConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)  # 嵌入層
        self.mha = MultiHeadAttention(cfg.d_model, cfg.num_heads, attn_dropout=cfg.dropout, resid_dropout=cfg.dropout)  # 多頭注意力層
        self.norm = nn.LayerNorm(cfg.d_model)  # LayerNorm
        self.cls = nn.Linear(cfg.d_model, 2)  # 分類層

    def forward(self, x_ids: torch.Tensor, pe: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.emb(x_ids)  # 將輸入轉換為嵌入
        if pe is not None:
            h = h + pe[:, : h.size(1), :].to(h.dtype)  # 加入位置編碼
        y, attn = self.mha(h, h, h, need_weights=True)  # 多頭注意力
        h = self.norm(h + y)  # 殘差連接 + LayerNorm
        pooled = h.mean(dim=1)  # 平均池化
        return self.cls(pooled), attn  # 返回分類結果和注意力權重

# 訓練一個 epoch 的函數
def train_one(model: nn.Module, 
              loader: DataLoader, 
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              pe:Optional[torch.Tensor]) -> tuple[float,float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss , total_correct, total = 0.0, 0, 0
    for xb ,yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb,pe=pe)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, total_correct / total

# 評估模型的函數
@torch.no_grad()
def eval_one(model: nn.Module, 
              loader: DataLoader,
              device: torch.device,
              pe:Optional[torch.Tensor]) -> tuple[float,float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss , total_correct, total = 0.0, 0, 0
    for xb ,yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb,pe=pe)
        loss = loss_fn(logits, yb)

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, total_correct / total

"""
作業要求
參考main()改寫 run_tiny_task_student_version 函式，讓使用者可以透過命令列參數設定以下選項：
1. 支援'="_ends_equal"'與'"compare_ij"'兩種任務類型。
2. 有參數'used_pe:bool'決定是否使用位置編碼。
3. 若 used_pe 為 True，則使用get_positional_encoding 函式產生 sin/cos 位置編碼，每一個batch都把正確長度的pe 傳入模型。
4. 若 used_pe 為 False，則不使用位置編碼。
"""
# 學生版本的 tiny task 主函數
def run_tiny_task_student_version(
    use_pe: bool = False,
    task: Literal["ends_equal", "compare_ij"] = "ends_equal",
    model_cfg: TinyConfig = TinyConfig(),
    seq_len: int = 12,
    n_samples: int = 6000,
    batch_size: int = 128,
    n_epochs: int = 10,
    device: Optional[torch.device] = None,
    seed: int = 1337,
    i: int = 0,
    j: int = -1,
    vocab_size: int = 100,
    lr: float = 3e-3,
    heads: int =4,
    d_model: int =128,
):
    set_seed(seed)  # 設定隨機種子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L = 12
    vocab = vocab_size
    n = n_samples
    j = j if j >=0 else L - 1

    # 根據任務類型生成數據集
    if task == "ends_equal":
        X, y = make_dataset_ends_equal(n, L, vocab)
    else:
        X, y = make_dataset_compare_ij(n, L, vocab, i= i, j= j)
    
    # 切分訓練集和驗證集
    n_train = int(n * 0.8)
    Xtr , ytr = X[:n_train], y[:n_train]
    Xva , yva = X[n_train:], y[n_train:]

    train_loader = DataLoader(torch.utils.data.TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.TensorDataset(Xva, yva), batch_size=batch_size)

    cfg = TinyConfig(
        vocab=vocab,
        d_model=128,
        num_heads=4,
        dropout=0,
    )
    model = TinyEncoder(cfg).to(device)

    pe = None
    if use_pe == True:
        pe = get_positional_encoding( max_len=L, d_model=d_model, device=device)
        # print("Using positional encoding.")

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"[Info] task = {task},  pe={use_pe}, L={L}, n={n}, d_model={d_model}, heads={heads},")
    for ep in range(1, n_epochs + 1):
        train_loss, train_acc = train_one(model, train_loader, optim, device, pe)
        val_loss, val_acc = eval_one(model, val_loader, device, pe)
        print(f"Epoch {ep:03d}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

# 主函數，解析命令列參數並執行任務
def main():
    parser = argparse.ArgumentParser(description="Tiny tasks for MHA + Position Encoding")
    # 設定命令列參數
    parser.add_argument("--task", type=str, choices=["ends_equal", "compare_ij"], default="ends_equal", help="Task type")
    parser.add_argument("--i", type=int, default=0, help="Index i for compare_ij task")
    parser.add_argument("--j", type=int, default=-1, help="Index j for compare_ij task")
    parser.add_argument("--len", type=int, default=12, help="Sequence length")
    parser.add_argument("--n", type=int, default=6000, help="Number of samples")
    parser.add_argument("--vocab", type=int, default=100, help="Vocabulary size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--pe", type=str, choices=["none", "sincos"], default="none", help="Type of positional encoding")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L = args.len
    vocab = args.vocab
    n = args.n
    j = args.j if args.j >=0 else L - 1

    # 根據任務類型生成數據集
    if args.task == "ends_equal":
        X, y = make_dataset_ends_equal(n, L, vocab)
    else:
        X, y = make_dataset_compare_ij(n, L, vocab, i= args.i, j= j)
    
    # 切分訓練集和驗證集
    n_train = int(n * 0.8)
    Xtr , ytr = X[:n_train], y[:n_train]
    Xva , yva = X[n_train:], y[n_train:]

    train_loader = DataLoader(torch.utils.data.TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.TensorDataset(Xva, yva), batch_size=args.batch_size)

    cfg = TinyConfig(
        vocab=vocab,
        d_model=args.d_model,
        num_heads=args.heads,
        dropout=args.dropout,
    )
    model = TinyEncoder(cfg).to(device)

    pe = None
    if args.pe == "sincos":
        pe = get_positional_encoding( max_len=L, d_model=args.d_model, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"[Info] task = {args.task},  pe={args.pe}, L={L}, n={n}, d_model={args.d_model}, heads={args.heads},")
    for ep in range(1, args.epochs + 1):
        train_loss, train_acc = train_one(model, train_loader, optim, device, pe)
        val_loss, val_acc = eval_one(model, val_loader, device, pe)
        print(f"Epoch {ep:03d}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

if __name__ == "__main__":
    print("========= 無位置編碼 =========")
    run_tiny_task_student_version(task="ends_equal", use_pe=False)
    print("========= 有位置編碼 =========")
    run_tiny_task_student_version(task="ends_equal", use_pe=True)
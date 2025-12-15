from __future__ import annotations
import math
from typing import Optional, Tuple, Literal
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import argparse
import random

# 嘗試匯入自定義模組，若失敗則嘗試從當前目錄匯入
try:
    from .mha import MultiHeadAttention
    from .position_encoding import get_positional_encoding
except Exception:
    from mha import MultiHeadAttention
    from position_encoding import get_positional_encoding

# 設定隨機種子以確保結果可重現
def set_seed(seed: int = 1337):
    random.seed(seed)  # 設定 Python 的隨機種子
    torch.manual_seed(seed)  # 設定 PyTorch 的隨機種子
    torch.cuda.manual_seed_all(seed)  # 設定所有 GPU 的隨機種子

# 建立一個序列資料集，輸入與輸出序列的開頭與結尾相同
def make_dataset_ends_equal(n: int, L: int, vocab: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """建立一個長度為 L 的序列資料集，輸入與輸出序列的開頭與結尾相同。
    Args:
        n: 資料集大小
        L: 序列長度
        vocab: 詞彙表大小
    Returns:
        X: 輸入序列
        y: 標籤 (1 表示開頭與結尾相同，0 表示不同)
    """
    X = torch.randint(1, vocab, (n, L))  # 隨機生成序列
    y = torch.zeros(n, dtype=torch.long)  # 初始化標籤
    half = n // 2

    # 前半部分的序列，開頭與結尾相同
    X[:half, -1] = X[:half, 0]
    y[:half] = 1

    # 後半部分的序列，開頭與結尾不同
    for i in range(half, n):
        a = X[i, 0].item()
        b = random.randint(1, vocab - 2)
        if b >= a:
            b += 1
        X[i, -1] = b
        y[i] = 0

    idx = torch.randperm(n)  # 隨機打亂資料
    return X[idx], y[idx]

# 建立一個序列資料集，輸入與輸出序列的第 i 與 j 個位置相同
def make_dataset_dompare_ij(n: int, L: int, vocab: int, i: int, j: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """建立一個長度為 L 的序列資料集，輸入與輸出序列的第 i 與 j 個位置相同。
    Args:
        n: 資料集大小
        L: 序列長度
        vocab: 詞彙表大小
        i: 第 i 個位置
        j: 第 j 個位置
    Returns:
        X: 輸入序列
        y: 標籤 (1 表示第 i 個位置大於第 j 個位置，0 表示否)
    """
    assert 1 <= i < L and 0 <= j < L and i != j, "i 和 j 的位置不合法"
    X = torch.randint(0, vocab, (n, L))  # 隨機生成序列
    y = (X[:, i] > X[:, j]).long()  # 比較第 i 和第 j 個位置的值
    return X, y

# 定義模型的配置參數
@dataclass
class TinyConfig:
    vocab: int = 100  # 詞彙表大小
    d_model: int = 128  # 嵌入維度
    num_heads: int = 4  # 注意力頭數
    dropout: float = 0.0  # Dropout 機率

# 定義 TinyEncoder 模型
class TinyEncoder(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)  # 詞嵌入層
        self.mha = MultiHeadAttention(cfg.d_model, cfg.num_heads, attn_dropout=cfg.dropout, resid_dropout=cfg.dropout)  # 多頭注意力層
        self.norm = nn.LayerNorm(cfg.d_model)  # LayerNorm 層
        self.cls = nn.Linear(cfg.d_model, 2)  # 分類層

    def forward(self, x_ids: torch.Tensor, pe: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.emb(x_ids)  # 將輸入序列轉換為嵌入
        if pe is not None:
            h = h + pe[:, : h.size(1), :].to(h.dtype)  # 加上位置編碼
        y, attn = self.mha(h, h, h, need_weights=True)  # 計算注意力
        h = self.norm(h + y)  # 殘差連接後進行正則化
        pooled = h.mean(dim=1)  # 平均池化 (B, D)
        return self.cls(pooled), attn  # 返回分類結果和注意力權重

# 訓練一個 epoch
def train_one(model: nn.Module, 
              loader: DataLoader, 
              optimizer: torch.optim.Optimizer,
              device: torch.device) -> tuple[float, float]:
    """訓練模型一個 epoch。
    Args:
        model: 模型
        loader: 資料加載器
        optimizer: 優化器
        device: 訓練設備
    Returns:
        平均損失和準確率
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()  # 定義交叉熵損失函數
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)  # 將資料移動到設備
        logits, _ = model(xb)  # 前向傳播
        loss = loss_fn(logits, yb)  # 計算損失

        optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 反向傳播
        optimizer.step()  # 更新參數

        total_loss += loss.item() * xb.size(0)  # 累加損失
        preds = logits.argmax(dim=-1)  # 預測類別
        total_correct += (preds == yb).sum().item()  # 計算正確數量
        total += xb.size(0)  # 總樣本數
    return total_loss / total, total_correct / total

# 評估模型
def eval_one(model: nn.Module, 
              loader: DataLoader,
              device: torch.device) -> tuple[float, float]:
    """評估模型。
    Args:
        model: 模型
        loader: 資料加載器
        device: 訓練設備
    Returns:
        平均損失和準確率
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()  # 定義交叉熵損失函數
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)  # 將資料移動到設備
        logits, _ = model(xb)  # 前向傳播
        loss = loss_fn(logits, yb)  # 計算損失

        total_loss += loss.item() * xb.size(0)  # 累加損失
        preds = logits.argmax(dim=-1)  # 預測類別
        total_correct += (preds == yb).sum().item()  # 計算正確數量
        total += xb.size(0)  # 總樣本數
    return total_loss / total, total_correct / total

# 主函數
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

    set_seed(args.seed)  # 設定隨機種子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 設定設備

    L = args.len
    vocab = args.vocab
    n = args.n
    j = args.j if args.j >= 0 else L - 1

    # 根據任務類型生成資料集
    if args.task == "ends_equal":
        X, y = make_dataset_ends_equal(n, L, vocab)
    else:
        X, y = make_dataset_dompare_ij(n, L, vocab, i=args.i, j=j)

    # 切分訓練集與驗證集
    n_train = int(n * 0.8)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:], y[n_train:]

    train_loader = DataLoader(torch.utils.data.TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.TensorDataset(Xva, yva), batch_size=args.batch_size)

    # 初始化模型與配置
    cfg = TinyConfig(
        vocab=vocab,
        d_model=args.d_model,
        num_heads=args.heads,
        dropout=args.dropout,
    )
    model = TinyEncoder(cfg).to(device)

    # 設定位置編碼
    pe = None
    if args.pe == "sincos":
        pe = get_positional_encoding(max_len=L, d_model=args.d_model, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 使用 AdamW 優化器

    print(f"[Info] task = {args.task},  pe={args.pe}, L={L}, n={n}, d_model={args.d_model}, heads={args.heads},")
    for ep in range(1, args.epochs + 1):
        train_loss, train_acc = train_one(model, train_loader, optim, device)  # 訓練一個 epoch
        val_loss, val_acc = eval_one(model, val_loader, device)  # 評估模型
        print(f"Epoch {ep:03d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

if __name__ == "__main__":
    main()
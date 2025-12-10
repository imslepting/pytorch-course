import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    from mamba_ssm import Mamba
except ImportError:
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state, d_conv, expand):
            super().__init__()
            # 假設如果 Mamba 無法導入，使用 TransformerEncoderLayer 作為替代
            # 注意：這是一個簡化或佔位符，實際的 Mamba 模型應該被正確導入
            self.layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                batch_first=True, # 輸入維度 [batch, seq, d_model]，跟你原本模型一致
            )
        def forward(self, x):
            return self.layer(x)

import numpy as np
from tqdm import tqdm # 用於顯示進度條

# --- MambaLanguageModel Class ---
class MambaLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Mamba 堆疊層
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for _ in range(n_layers)
        ])
        
        # 最終的 Layer Norm
        self.norm = nn.LayerNorm(d_model)
        # 語言模型頭 (預測下一個 token)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x = self.embedding(input_ids) 
        
        # 經過 Mamba 層
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)
            
        # Layer Norm
        x = self.norm(x)
        
        # LM Head -> [batch_size, seq_len, vocab_size]
        logits = self.lm_head(x) 
        
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None):
        self.eval() # 設置為評估模式
        
        for _ in range(max_length - input_ids.shape[1]):
            # 獲取最後一個 token 的 logits
            # [batch_size, seq_len, vocab_size]
            logits = self.forward(input_ids)[:, -1, :] 
            
            # 應用 Temperature
            next_token_logits = logits / temperature
            
            # Top-K 採樣
            if top_k is not None:
                # 獲取 Top-K 的閾值
                v, _ = torch.topk(next_token_logits, top_k)
                # 獲取最小的 Top-K logit 作為閾值
                # [batch_size, 1]
                min_val_of_topk = v[:, [-1]] 
                
                # 設置不在 Top-K 內的 logits 為負無窮大
                indices_to_remove = next_token_logits < min_val_of_topk
                next_token_logits[indices_to_remove] = float('-inf')

            # 轉為概率分佈
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # 根據概率採樣下一個 token
            # [batch_size, 1]
            next_token = torch.multinomial(probs, num_samples=1) 
            
            # 將新生成的 token 加入到 input_ids
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            
        return input_ids


# --- SimpleTextDataset Class ---
class SimpleTextDataset(Dataset):
    def __init__(self, texts, vocab, seq_length=64):
        self.texts = texts
        self.vocab = vocab
        self.seq_length = seq_length
        self.full_text = "".join(texts)
    
    def __len__(self):
        # 確保有足夠的長度來提取序列
        return max(0, len(self.full_text) - self.seq_length)

    def __getitem__(self, idx):
        # 提取長度為 seq_length + 1 的序列，用於 input (seq_length) 和 target (seq_length)
        # 例如 seq_length=32，則提取長度為 33 的子串
        seq = self.full_text[idx:idx + self.seq_length + 1]
        
        # input_ids: [c1, c2, ..., cN] (長度 N=seq_length)
        input_ids = torch.tensor(
            [self.vocab.get(c, 0) for c in seq[:-1]], # 排除最後一個
            dtype=torch.long
        )
        
        # target_ids: [c2, c3, ..., cN+1] (長度 N=seq_length)
        # 這是 input_ids 往後偏移一位的結果，即預測的目標
        target_ids = torch.tensor(
            [self.vocab.get(c, 0) for c in seq[1:]], # 排除第一個
            dtype=torch.long
        )
        
        return input_ids, target_ids


# --- Training and Utility Functions ---

def train_epoch(model, dataloader, optimizer, device):
    model.train() # 設置為訓練模式
    total_loss = 0.0
    
    # 遍歷資料加載器
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        # 將資料移動到指定設備
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # 前向傳播
        logits = model(input_ids)

        # 計算損失 (Cross Entropy Loss)
        # PyTorch 的 cross_entropy 期望 logits 的形狀為 [N, C, *] 和 target 的形狀為 [N, *]
        # view(-1, ...) 將 [batch, seq, vocab] 轉為 [batch*seq, vocab]
        loss = nn.functional.cross_entropy(
            logits.view(-1, model.vocab_size),
            target_ids.view(-1)
        )

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累積總損失
        total_loss += loss.item()

        # 打印進度 (可選)
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1:3d} | Loss: {loss.item():.4f}")

    # 返回平均損失
    avg_loss = total_loss / len(dataloader)
    return avg_loss


# --- Main Execution Block ---
if __name__ == "__main__":
    
    # 1. 設備設置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"f使用設備: {device}")

    # 2. 資料準備
    sample_text = "the quick brown fox jumps over the lazy dog."  
    chars = sorted(list(set(sample_text))) # 獲取所有唯一字元
    vocab = {c: i for i, c in enumerate(chars)} # 建立 char -> index 的映射
    vocab_size = len(vocab)
    
    print(f"\n字典大小: {vocab_size}")
    print(f"字元集: {chars}")
    
    # 3. 數據集和數據加載器
    texts = [sample_text *10 ] # 文本列表
    seq_length = 32 # 序列長度
    dataset = SimpleTextDataset(texts, vocab, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"資料集大小: {len(dataset)}")
    
    # 4. 模型初始化
    model = MambaLanguageModel(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
    )
    # 將模型移動到指定設備
    model.to(device) 
    
    # 計算模型參數數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數數量: {total_params}")

    # 5. 優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 6. 訓練循環
    print(f"\n開始訓練...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        print(f"平均損失: {avg_loss:.4f}")

    # 7. 文本生成示例
    print(f"\n文本生成範例:")
    model.eval() # 設置為評估模式
    
    # 初始化一個起始序列 (例如：從第一個字元開始)
    start_char = sample_text[0]
    start_id = vocab[start_char]
    # [batch_size=1, seq_len=1]
    initial_input = torch.tensor([[start_id]], dtype=torch.long).to(device) 
    
    # 生成新的文本
    generated_ids = model.generate(
        initial_input, 
        max_length=50, 
        temperature=0.8, 
        top_k=5
    )
    
    # 將生成的 token IDs 轉回字元
    id_to_char = {i: c for c, i in vocab.items()}
    generated_text = "".join([id_to_char[id.item()] for id in generated_ids.squeeze(0)])
    
    print(f"起始字元: '{start_char}'")
    print(f"生成文本: {generated_text}")
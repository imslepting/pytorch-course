import math
from typing import Optional, Tuple
import torch
import numpy as np

# 確保正確匯入 torch.nn.functional
import torch.nn.functional as F
TORCH_AVAILABLE = True

# 定義縮放點積注意力函數
# q: 查詢張量
# k: 鍵張量
# v: 值張量
# mask: 遮罩張量，可選
# dropout_p: dropout 機率
# 返回值: 包含輸出值和注意力權重的元組
def scaled_dot_product_attention(q, k, v, mask=None, dropout_p: float = 0.0):
    if TORCH_AVAILABLE:
        # 使用 PyTorch 實現
        d_k = q.size(-1)  # 查詢向量的最後一維大小
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # 計算注意力分數，並進行縮放
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 將遮罩位置設置為負無窮大
        attn_weights = F.softmax(scores, dim=-1)  # 計算注意力權重
        if dropout_p and dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)  # 應用 Dropout
        output = attn_weights @ v  # 計算輸出
        return output, attn_weights  # 返回輸出值和注意力權重
    else:
        # 使用 NumPy 實現
        d_k = q.shape[-1]  # 查詢向量的最後一維大小
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) / math.sqrt(d_k)  # 計算注意力分數
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)  # 將遮罩位置設置為負無窮大
        score_max = scores.max(axis=-1, keepdims=True)  # 計算每行的最大值，用於穩定 softmax
        exp = np.exp(scores - score_max)  # 計算指數值
        attn_weights = exp / exp.sum(axis=-1, keepdims=True)  # 計算注意力權重
        output = np.matmul(attn_weights, v)  # 計算輸出
        return output, attn_weights

# 建立填充遮罩
# lengths: 每個序列的有效長度
# max_len: 序列的最大長度
# 返回值: 布爾遮罩張量
def build_padding_mask(lengths, max_len):
    try:
        # 使用 PyTorch 實現
        device = lengths.device if hasattr(lengths, 'device') else None
        rng = torch.arange(max_len, device=device).unsqueeze(0)  # 生成範圍張量
        mask = (rng < lengths.unsqueeze(1)).int().to(torch.bool)  # 建立布爾遮罩
    except Exception:
        # 使用 NumPy 實現
        lengths = np.asarray(lengths)
        rng = np.arange(max_len)[None, :]  # 生成範圍陣列
        mask = (rng < lengths[:, None])  # 建立布爾遮罩
    return mask[:, None, None, :]

# 建立前瞻遮罩
# seq_len: 序列長度
# 返回值: 前瞻遮罩張量
def build_look_ahead_mask(seq_len):
    try:
        # 使用 PyTorch 實現
        return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))  # 下三角矩陣
    except Exception:
        # 使用 NumPy 實現
        m = np.tril(np.ones((seq_len, seq_len), dtype=bool))  # 下三角矩陣
        return m[None, None, :, :]

if __name__ == "__main__":
    print("torch available: ", TORCH_AVAILABLE)
    try:
        # 測試 PyTorch 實現
        torch.manual_seed(7)
        B, H, L, D = 2, 2, 5, 4  # 批次大小、注意力頭數、序列長度、嵌入維度

        q = torch.randn(B, H, L, D)  # 查詢張量
        k = torch.randn(B, H, L, D)  # 鍵張量
        v = torch.randn(B, H, L, D)  # 值張量

        output, attn = scaled_dot_product_attention(q, k, v, mask=None)

        print("[No mask] output shape: ", tuple(output.shape))
        print("[No mask] attn shape: ", tuple(attn.shape))
        print("Row sums (應為1) :", attn[0, 0, 0, :].sum().item())

        lengths = torch.tensor([3, 5])
        pad_mask = build_padding_mask(lengths, max_len=L)
        _, attn_padded = scaled_dot_product_attention(q, k, v, mask=pad_mask)
        print("\n[Padding mask] attn [0,0,0]: ", attn_padded[0, 0, 0])
        print("check pad columns (index>=3) : ", attn_padded[0, 0, 0, 3:].sum().item())

        look_mask = build_look_ahead_mask(L)
        la = torch.tensor(look_mask).expand(B, H, L, L)  # 修正 expand 方法
        _, attn_la = scaled_dot_product_attention(q, k, v, mask=la)
        print("\n[Look-ahead mask] 上三角是否為0? (印第一列): ", attn_la[0, 0, 0])
        print("整個attn矩陣:\n", attn_la[0, 0])

        scores = torch.tensor([[1.0, 0.0, -1.0],
                                [0.0, 1.0, 0.0],
                                [1.0, 1.0, 1.0]])
        weights = torch.softmax(scores, dim=-1)
        print("3x3 softmax:\n", weights)

    except Exception as e:
        # 測試 NumPy 實現
        np.random.seed(7)
        B, H, L, D = 2, 2, 5, 4

        q = np.random.randn(B, H, L, D)  # 查詢張量
        k = np.random.randn(B, H, L, D)  # 鍵張量
        v = np.random.randn(B, H, L, D)  # 值張量

        out, attn = scaled_dot_product_attention(q, k, v, mask=None)

        print("[No mask][NP] output shape: ", tuple(out.shape))
        print("[No mask][NP] attn shape: ", tuple(attn.shape))
        print("Row sums (應為1) :", attn[0, 0, 0].sum())

        lengths = np.array([3, 5])
        pad_mask = build_padding_mask(lengths, max_len=L)
        _, attn_padded = scaled_dot_product_attention(q, k, v, mask=pad_mask)
        print("[Padding mask][NP] attn [0,0,0]: ", attn_padded[0, 0, 0])
        print("check pad columns (index>=3) 近似0?: ", attn_padded[0, 0, 0, 3:].sum())

        look_mask = build_look_ahead_mask(L)
        la = np.broadcast_to(look_mask, (B, H, L, L))  # 修正 expand 方法
        _, attn_la = scaled_dot_product_attention(q, k, v, mask=la)
        print("[Look-ahead mask] 上三角是否為0? (印第一列): ", attn_la[0, 0, 0])

        scores = np.array([[1.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0]])
        scores = scores - np.max(scores, axis=-1, keepdims=True)  # 修正 max 方法
        exp = np.exp(scores)
        weights = exp / exp.sum(axis=-1, keepdims=True)
        print("3x3 softmax:\n", weights)
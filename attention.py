from __future__ import annotations
import math 
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = [
    "scaled_dot_product_attention",
    "make_padding_mask",
    "make_causal_mask",
    "combine_bool_masks",
]

# 用於生成與分數張量形狀相同的遮罩，將遮罩中的True值替換為負無窮大
# 這在計算注意力分數時非常有用，因為負無窮大會使softmax輸出接近於零
# scores: 注意力分數張量
# mask: 遮罩張量，必須是布爾類型
def _addiive_mask_like(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise ValueError(f"Expected mask dtype to be torch.bool, but got {mask.dtype}")
    neg_inf = torch.finfo(scores.dtype).min
    return torch.zeros(1,dtype=scores.dtype,device=scores.device).masked_fill(
        torch.tensor(True, device=scores.device), neg_inf) * mask.to(dtype=scores.dtype)

# 實現縮放點積注意力機制
# q: 查詢張量
# k: 鍵張量
# v: 值張量
# mask: 遮罩張量，可選
# dropout_p: dropout概率
# training: 是否處於訓練模式
# return_weights: 是否返回注意力權重
def scaled_dot_product_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask=None , dropout_p:float=0.0, training:bool=True , return_weights:bool=False):
    B,H,L,DK = q.shape
    scale = 1.0 / math.sqrt(DK)
    
    # 計算注意力分數
    scores = q @ k.transpose(-2, -1) / math.sqrt(DK)

    # 如果提供了遮罩，則應用遮罩
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask , torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask

    # 計算softmax以獲得注意力權重
    attn = F.softmax(scores, dim=-1)
    if dropout_p and training:
        attn = F.dropout(attn, p=dropout_p)
    # 計算最終的注意力輸出
    output = torch.matmul(attn, v)
    if return_weights:
        return output , attn
    return output

# 創建填充遮罩，用於標記序列中無效的位置
# lengths: 每個序列的有效長度
# L: 序列的最大長度
def make_padding_mask(lengths: torch.Tensor , L:int) -> torch.Tensor: # lengths = torch.tensor([3, 5]) 表示第一個序列長度為 3，第二個序列長度為 5。
    device = lengths.device 
    #先生成一個範圍張量 [0, 1, 2, ..., L-1] , 然後將形狀變為 (1, L) -> [[0, 1, 2, 3, 4]]
    idxs = torch.arange(L, device=device).unsqueeze(0)
    mask = idxs >= lengths.unsqueeze(1) # [3, 5] -> [[3], [5]] -> [[False, False, False, True, True], [False, False, False, False, False]]
    return mask.unsqueeze(1).unsqueeze(2) #　(B, L)　-> (B, 1, L) -> (B, 1, 1, L)

# 創建因果遮罩，用於防止訪問未來的信息
# L: 序列的長度
# device: 運算設備，可選
def make_causal_mask(L:int, device:Optional[torch.device]=None) -> torch.Tensor:
    mask = torch.triu(torch.ones((L,L), device=device, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0) #(L, L) -> (1, 1, L, L)

# 合併多個布爾遮罩
# masks: 多個布爾遮罩張量
def combine_bool_masks(*masks: torch.Tensor) -> torch.Tensor:
    valid = [m for m in masks if m is not None]
    if not valid:
        raise ValueError("At least one mask must be provided")
    out = valid[0]
    for m in valid[1:]:
        out = out | m
    return out

if __name__ == "__main__":
    # 測試縮放點積注意力的功能
    B,H,L,DK = 2,4,5,8
    q = torch.randn(B,H,L,DK)
    k = torch.randn(B,H,L,DK)
    v = torch.randn(B,H,L,DK)

    output, attn = scaled_dot_product_attention(q, k, v, return_weights=True)
    print("out:put shape: ", output.shape,"attn shape: ", attn.shape)
    print("q shape: ", q.shape)
    print("k shape: ", k.shape)
    print("v shape: ", v.shape)
    print("-----")
    print("Output shape: ", output.shape)
    print("Attent shape: ", attn.shape)
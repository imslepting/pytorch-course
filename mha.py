from __future__ import annotations
import math 
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# 嘗試從當前目錄匯入模組，若失敗則從當前路徑匯入
try:
    from .attention import(
        scaled_dot_product_attention,
        make_causal_mask,
        combine_bool_masks,
    )
except ImportError:
    from attention import(
        scaled_dot_product_attention,
        make_causal_mask,
        combine_bool_masks,
    )

# 定義多頭注意力機制類別
class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 512,  # 模型的維度
        num_heads: int = 8,  # 注意力頭的數量
        attn_dropout: float = 0.0,  # 注意力的 dropout 機率
        resid_dropout: float = 0.0,  # 殘差連接的 dropout 機率
        bias: bool = True,  # 是否使用偏置
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必須能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每個頭的維度

        # 定義查詢、鍵、值和輸出的線性層
        self.w_q = torch.nn.Linear(d_model, d_model, bias=bias)  # 查詢權重
        self.w_k = torch.nn.Linear(d_model, d_model, bias=bias)  # 鍵權重
        self.w_v = torch.nn.Linear(d_model, d_model, bias=bias)  # 值權重
        self.w_o = torch.nn.Linear(d_model, d_model, bias=bias)  # 輸出權重

        self.attn_dropout = attn_dropout
        self.resid_drop = nn.Dropout(resid_dropout)

    # 將輸入張量分割為多頭
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor: # (B,L,D) -> (B,H,L,D_k) 
        # D = H * D_k , L = 序列長度 , B = 批次大小
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        return x

    # 將多頭合併回單一張量
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor: # (B,H,L,D_k) -> (B,L,D)
        B, H, L, D_k = x.shape
        x = x.transpose(1, 2).contiguous().view(B, L, H * D_k)
        # transpose() 只改變張量的「視圖」，不改變底層記憶體佈局。
        # contiguous() 確保記憶體是連續的，這樣 view() 才能正確重塑張量！
        return x

    # 前向傳播函數
    def forward(
        self,
        q: torch.Tensor,  # 查詢張量
        k: torch.Tensor,  # 鍵張量
        v: torch.Tensor,  # 值張量
        *,
        attn_mask: Optional[torch.Tensor] = None,  # 注意力遮罩
        key_padding_mask: Optional[torch.Tensor] = None,  # 鍵的填充遮罩
        causal: bool = False,  # 是否使用因果遮罩
        need_weights: bool = True,  # 是否返回注意力權重
        average_attn_weights: bool = False,  # 是否平均注意力權重（拼寫錯誤修正為 average_attn_weights）
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = q.shape
        assert D == self.d_model and k.shape[-1] == D and v.shape[-1] == D, "查詢、鍵和值的維度必須與模型維度一致"

        # 計算查詢、鍵和值的投影
        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)

        # 分割多頭
        qh = self._split_heads(q_proj)
        kh = self._split_heads(k_proj)
        vh = self._split_heads(v_proj)

        # 構建遮罩
        mask_bool = None
        if key_padding_mask is not None:
            #(B,L) -> (B,1,1,L)
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(1)
            mask_bool = mask_k if mask_bool is None else (mask_bool | mask_k)
        if causal:
            mask_c = make_causal_mask(L, device=q.device)
            mask_bool = mask_c if mask_bool is None else (mask_bool | mask_c)

        # 確保傳入 combine_bool_masks 的參數不為 None
        if mask_bool is not None and attn_mask is not None:
            merged_mask = combine_bool_masks(mask_bool, attn_mask)
        elif mask_bool is not None:
            merged_mask = mask_bool
        elif attn_mask is not None:
            merged_mask = attn_mask
        else:
            merged_mask = None

        # 計算縮放點積注意力
        out_h, attn = scaled_dot_product_attention(
            qh, kh, vh,
            mask=merged_mask,
            dropout_p=self.attn_dropout,
            training=self.training,
            return_weights=True
        )

        # 合併多頭並通過輸出層
        out = self.w_o(self._merge_heads(out_h))
        out = self.resid_drop(out)

        if not need_weights:
            return out, None

        if average_attn_weights:  # 修正拼寫錯誤為 average_attn_weights
            attn_mean = attn.mean(dim=1)
            return out, attn_mean
        return out, attn

if __name__ == "__main__":
    # 測試多頭注意力機制
    B, L, D, H = 2, 10, 512, 8  # 批次大小、序列長度、維度、注意力頭數
    x = torch.randn(B, L, D)
    mha = MultiHeadAttention(d_model=D, num_heads=H)
    y, w = mha(x, x, x, causal=True, need_weights=True, average_attn_weights=True)  #  q k v = x
    print("y: ", y.shape, "\nw: ", w.shape)
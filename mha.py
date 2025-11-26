from __future__ import annotations
import math 
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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







class MultiHeadAttention(torch.nn.Module):





    def __init__(
        self,
        d_model: int = 512,
        num_heads:int = 8,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = torch.nn.Linear(d_model, d_model, bias=bias) #w_q  = query weight
        self.w_k = torch.nn.Linear(d_model, d_model, bias=bias)
        self.w_v = torch.nn.Linear(d_model, d_model, bias=bias)
        self.w_o = torch.nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = attn_dropout
        self.resid_drop = nn.Dropout(resid_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor: # (B,L,D) -> (B,H,L,D_k) 
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        return x
    



    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor: # (B,H,L,D_k) -> (B,L,D)
        B, H, L, D_k = x.shape
        x = x.transpose(1, 2).contiguous().view(B, L, H * D_k)
        return x
    



    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        need_weights: bool = True,
        average_attn_werghts: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = q.shape
        assert D == self.d_model and k.shape[-1] ==D and v.shape[-1] == D

        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)

        qh = self._split_heads(q_proj)
        kh = self._split_heads(k_proj)
        vh = self._split_heads(v_proj)

        mask_bool = None
        if key_padding_mask is not None:
            #(B,L) -> (B,1,1,L)
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(1)
            mask_bool = mask_k if mask_bool is None else ( mask_bool | mask_k)
        if causal:
            mask_c = make_causal_mask(L, device=q.device)
            mask_bool = mask_c if mask_bool is None else ( mask_bool | mask_c)

        merged_mask = None
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            merged_mask = combine_bool_masks(mask_bool, attn_mask)
        else:
            merged_mask = mask_bool if attn_mask is None else attn_mask

        out_h , attn = scaled_dot_product_attention(
            qh,kh,vh,
            mask = merged_mask,
            dropout_p=self.attn_dropout,
            training=self.training,
            return_weights=True
        )

        out = self.w_o(self._merge_heads(out_h))
        out = self.resid_drop(out)

        if not need_weights:
            return out, None
        
        if average_attn_werghts:
            attn_mean = attn.mean(dim=1)
            return out, attn_mean
        return out, attn
    
if __name__ == "__main__":
    B,L,D,H = 4,10,512,8 #Batch, Length, Dimension, Heads
    x = torch.randn(B,L,D)
    mha = MultiHeadAttention(d_model=D, num_heads=H)
    y,w = mha(x,x,x, causal=True, need_weights=True, average_attn_werghts=True)
    print("y: ", y.shape, "w: ", w.shape)
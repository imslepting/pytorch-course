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

def _addiive_mask_like(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    
    if mask.dtype != torch.bool:
        raise ValueError(f"Expected mask dtype to be torch.bool, but got {mask.dtype}")
    neg_inf = torch.finfo(scores.dtype).min
    return torch.zeros(1,dtype=scores.dtype,device=scores.device).masked_fill(
        torch.tensor(True, device=scores.device), neg_inf) * mask.to(dtype=scores.dtype)

def scaled_dot_product_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask=None , dropout_p:float=0.0, training:bool=True , return_weights:bool=False):
    B,H,L,DK = q.shape
    scale = 1.0 / math.sqrt(DK)
    
    scores = q @ k.transpose(-2, -1) / math.sqrt(DK)

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask , torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask

    attn = F.softmax(scores, dim=-1)
    if dropout_p and training:
        attn = F.dropout(attn, p=dropout_p)
    output = torch.matmul(attn, v)
    if return_weights:
        return output , attn
    return output

def make_padding_mask(lengths: torch.Tensor , L:int) -> torch.Tensor:
    device = lengths.device
    idxs = torch.arange(L, device=device).unsqueeze(0)
    mask = idxs >= lengths.unsqueeze(1)
    return mask.unsqueeze(1).unsqueeze(2)

def make_causal_mask(L:int, device:Optional[torch.device]=None) -> torch.Tensor:
    mask = torch.triu(torch.ones((L,L), device=device, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)

def combine_bool_masks(*masks: torch.Tensor) -> torch.Tensor:
    valid = [m for m in masks if m is not None]
    if not valid:
        raise ValueError("At least one mask must be provided")
    out = valid[0]
    for m in valid[1:]:
        out = out | m
    return out

if __name__ == "__main__":
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
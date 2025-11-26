import math
from typing import Optional, Tuple
import torch
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

def scaled_dot_product_attention(q,k,v,mask=None,dropout_p:float=0.0):
    if TORCH_AVAILABLE:
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        if dropout_p and dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        output = attn_weights @ v
        return output , attn_weights  # Output values and attention weights
    else:
        d_k = q.shape[-1]
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        score_max = scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores - score_max)
        attn_weights = exp / exp.sum(axis=-1, keepdims=True)
        output = np.matmul(attn_weights, v)
        return output , attn_weights
def build_padding_mask(lengths,max_len):
    try:
        device = lengths.device if hasattr(lengths, 'device') else None
        rng = torch.arange( max_len, device=device).unsqueeze(0)
        mask = (rng < lengths.unsqueeze(1)).int().to(torch.bool)
    except Exception:
        import numpy as np
        lengths = np.asarray(lengths)
        rng = np.arange(max_len)[None,:]
        mask = (rng < lengths[:,None])
        return mask[:,None,None,:]
    
def build_look_ahead_mask(seq_len):
    try:
        return torch.tril(torch.ones((seq_len,seq_len), dtype=torch.bool))
    except Exception:
        m = np.tril(np.ones((seq_len,seq_len), dtype=bool))
        return m[None,None,:,:]
    
if __name__ == "__main__":
    print("torch available: ", TORCH_AVAILABLE)
    try:
        torch.manual_seed(7)
        B,H,L,D = 2,2,5,4

        q = torch.randn(B, H, L, D) 
        k = torch.randn(B, H, L, D) 
        v = torch.randn(B, H, L, D) 

        output, attn = scaled_dot_product_attention(q, k, v , mask = None)

        print("[No mask] output shape: ", tuple(output.shape))
        print("[No mask] attn shape: ", tuple(attn.shape))
        print("Row sums (應為1) :" , attn[0,0,0,:].sum().item())

        lengths = torch.tensor([3,5])
        pad_mask = build_padding_mask(lengths, max_len=L)
        _,attn_padded = scaled_dot_product_attention(q, k, v , mask = pad_mask)
        print("\n[Padding mask] attn [0,0,0]: ", attn_padded[0,0,0])
        print("check pad columns (index>=3) : ", attn_padded[0,0,0,3:].sum().item())

        look_mask = build_look_ahead_mask(L)
        la,mask = look_mask.expand(B,H,L,L)
        _,attn_la = scaled_dot_product_attention(q, k, v , mask = la)
        print("\n[Look-ahead mask] 上三角是否為0? (印第一列): ", attn_la[0,0,0])
        print("整個attn矩陣:\n", attn_la[0,0])
        # print("--驗證batch0(長度為3)的attn矩陣","\n", attn_la[0,0,:3,:3])

        scores = torch.tensor([[1.0,0.0,-1.0],
                            [0.0,1.0,0.0],
                            [1.0,1.0,1.0]])
        weights = torch.softmax(scores, dim=-1)
        print("3x3 softmax:\n", weights)    

    except Exception as e:
        np.random.seed(7)
        B,H,L,D = 2,2,5,4

        q = torch.randn(B, H, L, D) 
        k = torch.randn(B, H, L, D) 
        v = torch.randn(B, H, L, D) 

        out, attn = scaled_dot_product_attention(q, k, v , mask = None)

        print("[No mask][NP] output shape: ", tuple(out.shape))
        print("[No mask][NP] attn shape: ", tuple(attn.shape))
        print("Row sums (應為1) :" , attn[0,0,0].sum().item())

        lengths = torch.tensor([3,5])
        pad_mask = build_padding_mask(lengths, max_len=L)
        _,attn_padded = scaled_dot_product_attention(q, k, v , mask = pad_mask)
        print("[Padding mask][NP] attn [0,0,0]: ", attn_padded[0,0,0])
        print("check pad columns (index>=3) 近似0?: ", attn_padded[0,0,0,3:].sum())

        look_mask = build_look_ahead_mask(L)
        la,mask = look_mask.expand(B,H,L,L)
        _,attn_la = scaled_dot_product_attention(q, k, v , mask = la)
        print("[Look-ahead mask] 上三角是否為0? (印第一列): ", attn_la[0,0,0])

        scores = torch.tensor([[1.0,0.0,-1.0],
                            [0.0,1.0,0.0],
                            [1.0,1.0,1.0]])
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores)
        weights = exp / exp.sum(axis=-1, keepdims=True)
        print("3x3 softmax:\n", weights)  
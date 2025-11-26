import torch
import torch.nn.functional as F
import math
import numpy as np

TORCH_AVAILABLE = True


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weight = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weight, v)
    return output , attn_weight  # Output values and attention weights


# def scaled_dot_product_attention(q, k, v, mask=None , dropout_p:float=0.0):
#     if TORCH_AVAILABLE:
#         d_k = q.size(-1)
#         scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#         attn_weight = F.softmax(scores, dim=-1)
#         if dropout_p and dropout_p > 0.0:
#             attn_weight = F.dropout(attn_weight, p=dropout_p)
#         output = torch.matmul(attn_weight, v)
#         return output , attn_weight  # Output values and attention weights
#     else:
#         d_k = q.size(-1)
#         scores = np.matmul(q, np.swqapaxes(k, -2, -1)) / math.sqrt(d_k)
#         if mask is not None:
#             scores = np.where(mask == 0, -1e9, scores)
#         score_max = scores.max(axis=-1, keepdims=True)
#         exp = np.exp(scores - score_max)
#         attn_weight = exp / exp.sum(axis=-1, keepdims=True)
#         output = np.matmul(attn_weight, v)
#         return output , attn_weight

BATCH_SIZE = 4
NUM_HEADS = 8 
SEQ_LEN = 10
D_K = 64
D_V = 64

q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K) 
k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K)
v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_V)

output, attn_weights = scaled_dot_product_attention(q, k, v)

print("q shape: ", q.shape)
print("k shape: ", k.shape)
print("v shape: ", v.shape)
print("-----")
print("Output shape: ", output.shape)
print("Attent shape: ", attn_weights.shape)
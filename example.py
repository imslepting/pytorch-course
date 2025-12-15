import torch
import torch.nn.functional as F
import math
import numpy as np

TORCH_AVAILABLE = True

# 实现缩放点积注意力机制
# q: 查询张量
# k: 键张量
# v: 值张量
# mask: 遮罩张量，可选
# 返回值: 包含输出值和注意力权重的元组
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)  # 获取查询张量的最后一维大小
    # 计算注意力分数，并进行缩放
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # 如果提供了遮罩，将遮罩中的无效位置设置为一个非常小的值
        scores = scores.masked_fill(mask == 0, -1e9)
    # 计算softmax以获得注意力权重
    attn_weight = F.softmax(scores, dim=-1)
    # 计算最终的注意力输出
    output = torch.matmul(attn_weight, v)
    return output , attn_weight  # 返回输出值和注意力权重

# 以下是另一个缩放点积注意力的实现版本，支持dropout和numpy运算
# 已被注释掉，仅供参考
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
#         return output , attn_weight  # 返回输出值和注意力权重
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

# 定义一些测试参数
BATCH_SIZE = 4  # 批次大小
NUM_HEADS = 8   # 注意力头的数量
SEQ_LEN = 10    # 序列长度
D_K = 64        # 查询和键的维度
D_V = 64        # 值的维度

# 随机生成查询、键和值张量
q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K) 
k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K)
v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_V)

# 调用缩放点积注意力函数
output, attn_weights = scaled_dot_product_attention(q, k, v)

# 输出张量的形状
print("q shape: ", q.shape)
print("k shape: ", k.shape)
print("v shape: ", v.shape)
print("-----")
print("Output shape: ", output.shape)
print("Attent shape: ", attn_weights.shape)
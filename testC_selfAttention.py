import torch , math
import torch.nn.functional as F

def sdp_attention(q,k,v,mask=None): 
    d = q.size(-1)
    scores = q @ k.transpose(-2,-1) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    w = F.softmax(scores, dim=-1)
    return w @ v , w # 輸出值與注意力權重

L , D = 16 , 2 # 序列長度與向量維度
q = torch.zeros(1,1,L,D)
k = torch.zeros(1,1,L,D)
v = torch.arange(L).float().view(1,1,L,1).repeat(1,1,1,D)

q[0,0,-1] = torch.tensor([1.0,0.0]) # 
k[0,0, 0] = torch.tensor([1.0,0.0])

for i in range(L-1):
    q[0,0,i] = torch.tensor([1.0,0.0])
    k[0,0,i] = torch.tensor([1.0,0.0])

out,attn = sdp_attention(q,k,v)
print("最後一列注意力權重(應該在位置0最高):", attn[0,0,-1])
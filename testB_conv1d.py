import torch,torch.nn as nn

L = 5 # 輸入序列長度
x = torch.zeros(1,1,L)
x[0,0,L//2] = 1.0 # 在序列中間放一個脈衝
layer = []
num_layers = 3
for _ in range(num_layers):
    conv = nn.Conv1d(1 , 1, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.ones_(conv.weight)
    layer.append(conv)
net = nn.Sequential(*layer)

with torch.no_grad():
    y = net(x)
    nz = (y[0,0] != 0).nonzero().flatten() # 計算非零元素的索引
    print(f'輸入 x : {x}')
    print(f'輸出 y : {y}')
    left , right = nz[0].item() , nz[-1].item() # 取得非零區間的左右邊界
    print(f'nz: {nz.tolist()}')
    print(f'非零區間跨度: [{right-left+1}] , 理論感受野 R_{num_layers}')
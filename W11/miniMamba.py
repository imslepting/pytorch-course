import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * self.weight
        return output
    
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4 ,expand=2):
        super().__init__()
        self.d_model = d_model # dimension of input and output
        self.d_state = d_state  # dimension of the SSM state
        self.d_conv = d_conv  # kernel size of the convolution
        self.expand = expand # expansion factor for the inner dimension
        self.d_inner = self.expand * self.d_model # dimension of the inner representation

        self.in_proj = nn.Linear(d_model, self.d_inner + 2 * self.d_state) # input projection

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding= d_conv -1
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_model *2, bias=False )
        self.dt_proj = nn.Linear(self.d_state, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype = torch.float32).repeat(self.d_inner,1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self,x):
        batch_size, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1,2) # (B,E,L) = 
        x = self.conv1d(x)[:, :, :seq_len] # 
        x = x.transpose(1,2) # (B,L,E)
        x = F.silu(x)

        y = self.ssm(x)
        y = self.norm(y)
        y = self.out_proj(y)

        return y
    
    def ssm(self, x):
        A = -torch.exp(self.A_log.float()) # (E,N)
        D = self.D.float()

        x_db1 = self.x_proj(x) # (B,L,dt_rank* + 2*N)
        delta, B, C = x_db1.split([self.d_state, self.d_model, self.d_model], dim=-1)

        delta = F.softplus(self.dt_proj(delta)) # (B,L,E)

        y = self.delective_scan(x, delta, A, B, C, D)
        return y
    
    def selective_scan(self, u, delta, A, B, C, D):
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        delta_A = torch.exp(delta.unsqueeze(-1) * A) # (B,L,E,N)
        delta_B_u = (delta.unsqueeze(-1) * B.unsqueeze(2)) * u.unsqueeze(-1) # (B,L,E,N)

        h = torch.zeros(batch_size, d_inner, d_state, device=u.device)# (B,E,N)
        ys = []

        for i in range(seq_len):
            h = delta_A[:,1]*h + delta_B_u[:,i]
            y = (h @ C[:,i,:].unsqueeze(-1)).squeeze(-1)# (B,E)
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + u*D

        return y

if __name__ == '__main__':
    d_model = 64
    batch_size = 2
    seq_len = 128
    

    model = MambaBlock(d_model=d_model)
    
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"x shape: {x.shape}")

    output = model(x)
    print(F"output shape: {output.shape}")

    assert output.shape == x.shape
    print("Output shape matches input shape.")

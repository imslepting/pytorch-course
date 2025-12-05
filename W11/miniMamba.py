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
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.d_inner = self.expand * self.d_model

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        # --- 修改點 1: x_proj 的輸出維度 ---
        # B 和 C 應該對應 d_state (16)，而不是 d_model (64)
        # 所以總輸出是: d_state(for delta) + d_state(for B) + d_state(for C)
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_state * 2, bias=False)
        
        # --- 修改點 2: dt_proj 的輸入輸出方向 ---
        # 這是將低維的 delta (d_state) 投影放大回 (d_inner)
        self.dt_proj = nn.Linear(self.d_state, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        # ... (forward 保持不變) ...
        batch_size, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(z)
        y = self.norm(y)
        y = self.out_proj(y)
        return y

    def ssm(self, x):
        A = -torch.exp(self.A_log.float())    # (d_inner, d_state)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)

        # --- 修改點 3: split 的維度 ---
        # 根據修改後的 x_proj，這裡要切成三個 d_state
        delta, B, C = x_dbl.split(
            [self.d_state, self.d_state, self.d_state], dim=-1
        )

        delta = F.softplus(self.dt_proj(delta))   # 現在這裡形狀正確了: (B, L, 16) -> (B, L, 128)

        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    # ... (selective_scan 保持不變) ...
    def selective_scan(self, u, delta, A, B, C, D):
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        delta_A = torch.exp(delta.unsqueeze(-1) * A)                  # (B, L, E, N)
        delta_B_u = (delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1))  # (B, L, E, N)

        h = torch.zeros(batch_size, d_inner, d_state, device=u.device)
        ys = []

        for i in range(seq_len):
            h = delta_A[:, i] * h + delta_B_u[:, i]
            y = (h @ C[:, i, :].unsqueeze(-1)).squeeze(-1)           # (B, E)
            ys.append(y)

        y = torch.stack(ys, dim=1)   # (B, L, E)
        y = y + u * D                # (B, L, E)

        return y
        
if __name__ == '__main__':
    d_model = 64
    batch_size = 2
    seq_len = 128
    

    model = MambaBlock(d_model=d_model)
    
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"x shape: {x.shape}")

    output = model(x)
    print(f"output shape: {output.shape}")

    assert output.shape == x.shape
    print("Output shape matches input shape.")

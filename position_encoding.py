from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn

# 定義模組的公開接口
__all__ = ["get_positional_encoding", "SunusoidalPositionEncoding"]

# 定義一個函數，用於生成位置編碼
def get_positional_encoding(
        max_len: int,  # 最大序列長度
        d_model: int,  # 嵌入維度
        device: Optional[torch.device] = None,  # 運算設備，可選
        dtype: Optional[torch.dtype] = None,  # 資料類型，可選
) -> torch.Tensor:
    """回傳形狀(1, max_len, d_model)的sin/cos編碼。"""
    if dtype is None:
        dtype = torch.get_default_dtype()  # 如果未指定資料類型，使用預設類型

    # 初始化位置編碼張量，形狀為 (max_len, d_model)
    pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)

    # 計算位置索引，形狀為 (max_len, 1)
    position = torch.arange(0, max_len, device=device, dtype=dtype).unsqueeze(1)

    # 計算除數項，形狀為 (d_model // 2)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model)
    )

    # 將 sin 和 cos 的值填入位置編碼張量
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶數維度使用 sin
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇數維度使用 cos

    # 增加批次維度，形狀變為 (1, max_len, d_model)
    pe = pe.unsqueeze(0)
    return pe

# 定義一個類別，用於生成並應用位置編碼
class SunusoidalPositionEncoding(nn.Module):
    def __init__(
            self,
            d_model: int,  # 嵌入維度
            max_len: int = 10000,  # 最大序列長度
            dropout: float = 0.0,  # dropout 機率
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)  # 定義 dropout 層

        # 使用 get_positional_encoding 函數生成位置編碼
        pe = get_positional_encoding(max_len, d_model)

        # 將位置編碼註冊為緩衝區，避免被當作模型參數
        self.register_buffer("pe", pe, persistent=False)

    # 前向傳播函數
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)  # 獲取輸入的序列長度
        # 將位置編碼加到輸入張量上
        x = x + self.pe[:, :L, :].to(dtype=x.dtype)
        return self.dropout(x)  # 應用 dropout 後返回

if __name__ == "__main__":
    # 測試位置編碼函數
    pe = get_positional_encoding(8, 6)
    print(pe.shape)  # 應輸出 torch.Size([1, 8, 6])
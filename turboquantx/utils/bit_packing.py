"""
位打包工具：高效压缩量化索引
"""

import torch
import numpy as np
from typing import Union


class BitPacker:
    """位打包器：将整数索引高效压缩为位流"""
    
    def __init__(self, bit_width: int):
        """
        Args:
            bit_width: 每个值的比特数
        """
        self.bit_width = bit_width
        self.max_value = (1 << bit_width) - 1
    
    def pack(self, indices: Union[torch.Tensor, np.ndarray]) -> bytes:
        """打包整数索引为字节流"""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        
        # 确保索引在有效范围内
        indices = np.clip(indices, 0, self.max_value).astype(np.uint32)
        
        # 计算需要的字节数
        total_bits = indices.size * self.bit_width
        total_bytes = (total_bits + 7) // 8
        
        # 创建输出缓冲区
        packed = bytearray(total_bytes)
        
        # 位打包
        bit_pos = 0
        for idx in indices.flat:
            # 将每个索引的比特写入缓冲区
            for bit in range(self.bit_width):
                byte_pos = bit_pos // 8
                bit_offset = bit_pos % 8
                
                if idx & (1 << bit):
                    packed[byte_pos] |= (1 << bit_offset)
                
                bit_pos += 1
        
        return bytes(packed)
    
    def unpack(self, packed_data: bytes, shape: tuple) -> np.ndarray:
        """从字节流解包为整数索引"""
        total_elements = np.prod(shape)
        total_bits = total_elements * self.bit_width
        
        # 创建输出数组
        indices = np.zeros(total_elements, dtype=np.uint32)
        
        # 位解包
        bit_pos = 0
        for i in range(total_elements):
            idx = 0
            for bit in range(self.bit_width):
                byte_pos = bit_pos // 8
                bit_offset = bit_pos % 8
                
                if byte_pos < len(packed_data) and (packed_data[byte_pos] >> bit_offset) & 1:
                    idx |= (1 << bit)
                
                bit_pos += 1
            
            indices[i] = idx
        
        return indices.reshape(shape)
    
    def compression_ratio(self, original_dtype: str = 'int32') -> float:
        """计算压缩比"""
        if original_dtype == 'int32':
            original_bits = 32
        elif original_dtype == 'int16':
            original_bits = 16
        else:
            original_bits = 32  # 默认
        
        return original_bits / self.bit_width
    
    def __repr__(self) -> str:
        return f"BitPacker(bit_width={self.bit_width}, max_value={self.max_value})"
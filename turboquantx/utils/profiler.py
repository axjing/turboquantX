"""
性能分析器：量化器性能监控和分析
"""

import time
import torch
from typing import Dict, List, Optional
from contextlib import contextmanager


class QuantizationProfiler:
    """量化性能分析器"""
    
    def __init__(self):
        self.records = {}
        self.current_record = None
    
    @contextmanager
    def record(self, name: str):
        """记录性能的上下文管理器"""
        start_time = time.time()
        start_memory = self._get_gpu_memory() if torch.cuda.is_available() else 0
        
        try:
            self.current_record = name
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_gpu_memory() if torch.cuda.is_available() else 0
            
            duration = end_time - start_time
            memory_used = end_memory - start_memory if end_memory > start_memory else 0
            
            if name not in self.records:
                self.records[name] = []
            
            self.records[name].append({
                'duration': duration,
                'memory_used': memory_used,
                'timestamp': time.time()
            })
            
            self.current_record = None
    
    def _get_gpu_memory(self) -> int:
        """获取GPU内存使用量"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return 0
    
    def get_stats(self, name: Optional[str] = None) -> Dict:
        """获取性能统计"""
        if name:
            if name not in self.records:
                return {}
            records = self.records[name]
        else:
            records = []
            for record_list in self.records.values():
                records.extend(record_list)
        
        if not records:
            return {}
        
        durations = [r['duration'] for r in records]
        memory_used = [r['memory_used'] for r in records]
        
        return {
            'count': len(records),
            'total_time': sum(durations),
            'avg_time': sum(durations) / len(durations),
            'min_time': min(durations),
            'max_time': max(durations),
            'total_memory': sum(memory_used),
            'avg_memory': sum(memory_used) / len(memory_used),
            'records': records if name else None
        }
    
    def reset(self):
        """重置记录"""
        self.records = {}
        self.current_record = None
    
    def print_summary(self):
        """打印性能摘要"""
        print("=== 量化性能摘要 ===")
        for name in self.records:
            stats = self.get_stats(name)
            print(f"{name}:")
            print(f"  调用次数: {stats['count']}")
            print(f"  平均时间: {stats['avg_time']:.6f}s")
            print(f"  总时间: {stats['total_time']:.6f}s")
            if stats['avg_memory'] > 0:
                print(f"  平均内存: {stats['avg_memory'] / 1024**2:.2f} MB")
            print()
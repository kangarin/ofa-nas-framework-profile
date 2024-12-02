'''
用来确定设备合理的batch size上限
'''

import torch
import torchvision.models as models
import time
import numpy as np
from typing import List, Tuple

def generate_random_input(batch_size: int, input_size : Tuple[int, int],
                          device: str) -> torch.Tensor:
    """生成随机输入数据"""
    return torch.randn(batch_size, 3, *input_size, device=device)

def measure_inference_time(model: torch.nn.Module, 
                         input_tensor: torch.Tensor,
                         device: str,
                         warmup: int = 10,
                         num_repeats: int = 100) -> Tuple[float, float]:
    """测量推理时延"""
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    # 测量时延
    times = []
    with torch.no_grad():
        for _ in range(num_repeats):
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000) 
    
    return np.mean(times), np.std(times)

def test_batch_sizes(model: torch.nn.Module, 
                    batch_sizes: List[int], 
                    input_size: Tuple[int, int],
                    device: str,
                    warmup: int = 10,
                    num_repeats: int = 100) -> List[Tuple[int, float, float]]:
    
    model = model.to(device)
    """测试不同batch size的性能"""
    results = []
    
    print(f"\nTesting on {device.upper()}:")
    print(f"{'Batch Size':^10} | {'Avg Latency (ms)':^14} | {'Std Dev (ms)':^11} | {'Throughput (samples/s)':^20}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        input_tensor = generate_random_input(batch_size, input_size, device)
        avg_time, std_time = measure_inference_time(model, input_tensor, device, warmup, num_repeats)
        throughput = (batch_size * 1000) / avg_time
        results.append((batch_size, avg_time, std_time))
        print(f"{batch_size:^10d} | {avg_time:^14.2f} | {std_time:^11.2f} | {throughput:^20.2f}")
    
    return results

if __name__ == "__main__":

    # 加载模型
    from torchvision.models import resnet50
    model = resnet50()

    input_size = (360, 640)

    batch_sizes = [8, 16]
    test_batch_sizes(model, batch_sizes, input_size, 'cuda', warmup=10, num_repeats=100)
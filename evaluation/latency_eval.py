import torch
import numpy as np

def eval_latency(model, input_size, device, warmup_times = 10, repeat_times = 100):
    import time
    model = model.to(device)
    model.eval()

    # 如果是cpu，只用一半的repeat_times
    if device == torch.device("cpu"):
        repeat_times = repeat_times // 2

    times = []
    
    with torch.no_grad():
        for i in range(warmup_times):
            img = torch.randn(1, 3, input_size, input_size).to(device)
            model(img)

        for i in range(repeat_times):
            img = torch.randn(1, 3, input_size, input_size).to(device)
            start = time.time()
            model(img)
            end = time.time()
            times.append(end - start)

        avg_time = np.mean(times)
        std_time = np.std(times)

    return avg_time, std_time


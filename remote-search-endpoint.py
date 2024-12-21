import asyncio
import websockets
import json
import torch
import numpy as np
import time
from models.detection.ofa_mbv3_w12_fasterrcnn import get_ofa_mbv3_w12_fasterrcnn_model

# 初始化模型
model = get_ofa_mbv3_w12_fasterrcnn_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

async def measure_latency(config, resolution, warmup_times=10, repeat_times=50):
    """测量延迟的函数"""
    model.backbone.body.set_active_subnet(**config)
    times = []
    
    with torch.no_grad():
        # 预热
        for _ in range(warmup_times):
            img = torch.randn(1, 3, resolution, resolution).to(device)
            model(img)

        # 正式测量
        for _ in range(repeat_times):
            img = torch.randn(1, 3, resolution, resolution).to(device)
            start = time.time()
            model(img)
            end = time.time()
            times.append(end - start)

    avg_time = float(np.mean(times))
    std_time = float(np.std(times))
    return avg_time, std_time

async def client_handler():
    """WebSocket客户端处理函数"""
    uri = "ws://localhost:8768"
    
    while True:  # 持续重连机制
        try:
            async with websockets.connect(uri) as websocket:
                print("Connected to server")
                
                while True:
                    # 等待服务器发送配置
                    message = await websocket.recv()
                    data = json.loads(message)
                    print("Received request:", data)
                    
                    # 解析配置
                    config = data['config']
                    resolution = data['resolution']
                    warmup_times = data.get('warmup_times', 10)
                    repeat_times = data.get('repeat_times', 50)
                    
                    # 执行测量
                    avg_time, std_time = await measure_latency(
                        config,
                        resolution,
                        warmup_times,
                        repeat_times
                    )
                    
                    # 发送结果
                    result = {
                        "latency": avg_time,
                        "std": std_time
                    }
                    await websocket.send(json.dumps(result))
                    print("Sent result:", result)
                    
        except websockets.ConnectionClosed:
            print("Connection lost. Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Error occurred: {e}")
            await asyncio.sleep(5)

if __name__ == '__main__':
    print("Starting client...")
    asyncio.get_event_loop().run_until_complete(client_handler())
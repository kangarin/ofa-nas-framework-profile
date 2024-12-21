from flask import Flask, request, jsonify
import torch
import numpy as np
import time

# 使用ofa-mbv3-w12-fasterrcnn模型
from models.detection.ofa_mbv3_w12_fasterrcnn import get_ofa_mbv3_w12_fasterrcnn_model
model = get_ofa_mbv3_w12_fasterrcnn_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

def measure_latency(model, input_size, device, warmup_times=10, repeat_times=100):
    model = model.to(device)
    model.eval()
    
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

@app.route('/measure_latency', methods=['POST'])
def measure_latency_endpoint():
    data = request.get_json()
    print(data)
    
    # 从请求中获取参数
    config = data['config']
    resolution = data['resolution']
    warmup_times = data.get('warmup_times', 10)
    repeat_times = data.get('repeat_times', 50)

    model.backbone.body.set_active_subnet(**config)
    
    avg_time, std_time = measure_latency(
        model,
        resolution,
        device,
        warmup_times,
        repeat_times
    )
    
    return jsonify({
        "latency": float(avg_time),
        "std": float(std_time)
    })

if __name__ == '__main__':
    # 这里需要初始化你的模型
    # model = get_model()
    app.run(host='0.0.0.0', port=5566)
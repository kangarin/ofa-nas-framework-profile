from optuna.samplers import NSGAIISampler
import numpy as np

class CustomNSGAIISampler(NSGAIISampler):
    def __init__(self, latency_weight=5.0, **kwargs):
        super().__init__(**kwargs)
        self.latency_weight = latency_weight
        
    def _calculate_crowding_distance(self, points, bounds):
        # points: numpy array of (n_points, n_objectives)
        n_points = len(points)
        n_objectives = len(points[0])
        
        crowding_distances = np.zeros(n_points)
        
        for i in range(n_objectives):
            weight = self.latency_weight if i == 1 else 1.0  # 假设第二个目标是延迟
            sorted_indices = np.argsort(points[:, i])
            # 确保边界点被保留
            crowding_distances[sorted_indices[0]] = float('inf')
            crowding_distances[sorted_indices[-1]] = float('inf')
            
            # 计算中间点的拥挤度距离
            norms = bounds[i][1] - bounds[i][0]
            for j in range(1, n_points - 1):
                prev_val = points[sorted_indices[j - 1], i]
                next_val = points[sorted_indices[j + 1], i]
                crowding_distances[sorted_indices[j]] += weight * (next_val - prev_val) / norms
                
        return crowding_distances
from models.backbone.ofa_supernet import get_architecture_dict

def get_best_arch_configs(study, backbone_name: str):
    """
    从study中提取pareto最优的trials并转换为架构配置
    
    Args:
        study: optuna.Study对象
        backbone_name: 如'ofa_supernet_mbv3_w10'等
        
    Returns:
        List[Dict]: 列表包含多个配置字典,每个字典格式如:
        {
            'values': [objective1, objective2],
            'resolution': int,
            'config': {
                'ks': [...],
                'e': [...], 
                'd': [...]
            }
        }
    """
    # 获取架构参数定义
    arch_dict = get_architecture_dict(backbone_name)
    
    # 获取pareto最优的trials
    best_trials = study.best_trials
    
    configs = []
    for trial in best_trials:
        # 提取目标值
        values = trial.values
        
        # 获取分辨率参数
        resolution_idx = trial.params['r']
        
        # 转换架构参数
        config = {}
        for param_name, param_info in arch_dict.items():
            length = param_info['length']
            choices = param_info['choices']
            
            # 收集该参数的所有值
            param_values = []
            for i in range(length):
                param_key = f'{param_name}{i+1}'
                idx = trial.params[param_key]
                actual_value = choices[idx]
                param_values.append(actual_value)
            
            config[param_name] = param_values
            
        configs.append({
            'values': values,
            'resolution_idx': resolution_idx,
            'config': config
        })
    
    return configs
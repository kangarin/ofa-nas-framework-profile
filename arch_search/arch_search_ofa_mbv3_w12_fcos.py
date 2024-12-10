import optuna
from models.backbone.ofa_supernet import get_architecture_dict
from optuna.samplers import NSGAIIISampler
from evaluation.detection_accuracy_eval import eval_accuracy
from evaluation.latency_eval import eval_latency
from datasets.calib_dataset import get_calib_dataset, create_fixed_size_dataloader
from datasets.common_transform import common_transform_list
from torchvision import transforms
from utils.bn_calibration import set_running_statistics
import torch

class ArchSearchOFAMbv3W12Fcos:
    def __init__(self, model, device, resolution_list):
        self.model = model
        self.device = device
        self.resolution_list = resolution_list
        self.calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_list))
        self.calib_dataloader = create_fixed_size_dataloader(self.calib_dataset, 10)

    def objective(self, trial):
        trial_number = trial.number
        arch_dict = get_architecture_dict('ofa_supernet_mbv3_w12')

        # 动态创建所有架构参数
        config = {}
        trial_params = {}

        for param_name, param_info in arch_dict.items():
            length = param_info['length']
            choices = param_info['choices']
            # 为每个参数创建trial建议值
            param_values = [
                trial.suggest_int(f'{param_name}{i+1}', 0, len(choices)-1) 
                for i in range(length)
            ]
            # 将索引映射到实际值
            mapped_values = [choices[idx] for idx in param_values]
            # 存储映射后的值
            config[param_name] = mapped_values
            trial_params[param_name] = param_values

        # 分辨率参数
        
        r = trial.suggest_int('r', 0, len(self.resolution_list)-1)
        r_values = self.resolution_list
        r_mapped = r_values[r]
        print("Arch: ", config, "resolution: ", r_mapped)

        objective1 = get_accuracy(self.model, config, r_mapped, self.calib_dataloader, self.device)
        objective2 = get_latency(self.model, config, r_mapped, self.device)

        return objective1, objective2
    
def get_accuracy(model, config, img_size, calib_dataloader, device):
    model.backbone.body.set_active_subnet(**config)
    set_running_statistics(model, calib_dataloader)
    # 对于精度，可以用gpu加速
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    result = eval_accuracy(model, img_size, 100, device, show_progress=True)
    return result['AP@0.5:0.95']

def get_latency(model, config, img_size, device):
    model.backbone.body.set_active_subnet(**config)
    result = eval_latency(model, img_size, device)
    return result[0]
    
def create_study(study_name):
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, 
                                storage=storage_name, 
                                directions=["maximize", "minimize"],
                                load_if_exists=True,
                                sampler=NSGAIIISampler())
    return study

def run_study(model, study, n_trials, device, resolution_list):
    arch_searcher = ArchSearchOFAMbv3W12Fcos(model, device, resolution_list)
    objective = arch_searcher.objective
    study.optimize(objective, n_trials=n_trials)

def plot_pareto_front(study):
    from optuna.visualization import plot_pareto_front
    plot_pareto_front(study)
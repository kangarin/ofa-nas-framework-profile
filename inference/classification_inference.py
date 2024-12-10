from datasets.calib_dataset import create_fixed_size_dataloader, get_calib_dataset
from utils.bn_calibration import set_running_statistics
import torch
from datasets.common_transform import common_transform_with_normalization_list
from torchvision import transforms

class ClassificationInference:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        model.to(device)
        model.eval()
        calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_with_normalization_list))
        self.calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
        set_running_statistics(model, self.calib_dataloader, 10)

    def set_active_subnet(self, **subnet_config):
        ofa_backbone_body = self.model.backbone.body
        ofa_backbone_body.set_active_subnet(**subnet_config)
        set_running_statistics(self.model, self.calib_dataloader, 10)

    def classify(self, image):
        with torch.no_grad():
            result = self.model(image.to(self.device))
        return result
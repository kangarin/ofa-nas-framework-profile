from datasets.calib_dataset import create_fixed_size_dataloader, get_calib_dataset
from utils.bn_calibration import set_running_statistics
import torch
from datasets.common_transform import common_transform_list
from torchvision import transforms

class DetectionInference:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        model.to(device)
        model.eval()
        calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_list))
        self.calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
        set_running_statistics(model, self.calib_dataloader, 10)

    def set_active_subnet(self, **subnet_config):
        ofa_backbone_body = self.model.backbone.body
        ofa_backbone_body.set_active_subnet(**subnet_config)
        set_running_statistics(self.model, self.calib_dataloader, 10)

    def detect(self, image, threshold=0.5):
        '''
        Args:
            image: torch.Tensor, shape [B, C, H, W]
            threshold: float, 置信度阈值
        Returns:
            batch_boxes: list of numpy arrays, 每个元素形状为 [N, 4]
            batch_labels: list of numpy arrays, 每个元素形状为 [N]
            batch_scores: list of numpy arrays, 每个元素形状为 [N]
            其中N为每张图片检测到的目标数量
        '''
        with torch.no_grad():
            predictions = self.model(image.to(self.device))
        
        batch_boxes = []
        batch_labels = []
        batch_scores = []
        
        # 处理每张图片的预测结果
        for prediction in predictions:
            # 过滤低置信度的预测结果
            mask = prediction['scores'] > threshold
            boxes = prediction['boxes'][mask].cpu().numpy()
            labels = prediction['labels'][mask].cpu().numpy()
            scores = prediction['scores'][mask].cpu().numpy()
            
            batch_boxes.append(boxes)
            batch_labels.append(labels)
            batch_scores.append(scores)
        
        return batch_boxes, batch_labels, batch_scores
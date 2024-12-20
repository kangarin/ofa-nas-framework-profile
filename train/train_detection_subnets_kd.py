from utils.bn_calibration import set_running_statistics
from datasets.calib_dataset import get_calib_dataset, create_fixed_size_dataloader
from datasets.coco_dataset import get_train_dataset, get_dataloader
import torch
from datasets.common_transform import common_transform_list
from torchvision import transforms
from utils.logger import setup_logger
from evaluation.detection_accuracy_eval import eval_accuracy
import random

logger = setup_logger('train')

def fpn_distill_loss(teacher_fpn, student_fpn):
    """计算FPN特征图的蒸馏损失"""
    dist_loss = 0
    for level in teacher_fpn.keys():
        t_feat = teacher_fpn[level]
        s_feat = student_fpn[level]
        
        # 计算attention map
        t_attention = torch.sum(torch.pow(t_feat, 2), dim=1, keepdim=True)
        s_attention = torch.sum(torch.pow(s_feat, 2), dim=1, keepdim=True)
        
        # normalize attention maps
        t_attention = torch.nn.functional.normalize(t_attention.view(t_attention.size(0), -1), dim=1).view_as(t_attention)
        s_attention = torch.nn.functional.normalize(s_attention.view(s_attention.size(0), -1), dim=1).view_as(s_attention)
        
        # attention guided feature distillation
        t_feat = t_feat * t_attention 
        s_feat = s_feat * s_attention
        
        # L2 loss
        dist_loss += torch.nn.functional.mse_loss(s_feat, t_feat)
    
    return dist_loss

def train_selected_subnets(model, subnet_configs, max_net_config, num_epochs, save_path,
                          batch_size=1,
                          backbone_learning_rate=1e-4, head_learning_rate=1e-4,
                          min_backbone_lr=1e-4, min_head_lr=1e-4,
                          subnet_sample_interval=5,
                          distill_alpha=5.0):
    """
    Train detection network using selected subnet configurations with knowledge distillation.
    
    Args:
        model: Model to train
        subnet_configs: List of subnet configurations to sample from
        max_net_config: Configuration for the largest network (teacher)
        num_epochs: Number of training epochs
        save_path: Path to save trained model
        batch_size: Batch size for training
        backbone_learning_rate: Initial learning rate for backbone
        head_learning_rate: Initial learning rate for detection head
        min_backbone_lr: Minimum learning rate for backbone
        min_head_lr: Minimum learning rate for detection head
        subnet_sample_interval: Interval for switching subnet configurations
        distill_alpha: Weight for distillation loss
    """
    # 设置优化器参数
    params_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if hasattr(model, 'head'):
        params_head = [p for p in model.head.parameters() if p.requires_grad]
    elif hasattr(model, 'roi_heads'):
        params_head = [p for p in model.roi_heads.parameters() if p.requires_grad] + \
                     [p for p in model.rpn.parameters() if p.requires_grad]

    params_backbone = [{'params': params_backbone, 'lr': backbone_learning_rate}]
    params_head = [{'params': params_head, 'lr': head_learning_rate}]
    
    optimizer_backbone = torch.optim.Adam(params_backbone, lr=backbone_learning_rate)
    optimizer_head = torch.optim.Adam(params_head, lr=head_learning_rate)
    
    scheduler_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_backbone, T_max=num_epochs, eta_min=min_backbone_lr, verbose=True
    )
    scheduler_head = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_head, T_max=num_epochs, eta_min=min_head_lr, verbose=True
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Start training, using device: {device}")
    model.to(device)

    calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_list))
    calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
    
    train_dataloader = get_dataloader(get_train_dataset(), batch_size)
    
    ofa_network = model.backbone.body
    current_config = None
    
    for epoch in range(num_epochs):
        model.train()
        i = 0
        loss_sum = 0
        
        for data in train_dataloader:
            if not data:
                continue
                
            if i > 1000:  # 临时用于本地测试
                break
                
            images, targets = data
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 每subnet_sample_interval次迭代切换一次网络配置
            if i % subnet_sample_interval == 0:
                current_config = random.choice(subnet_configs)
                ofa_network.set_active_subnet(**current_config)
                logger.info(f"Switched to subnet config: {current_config}")
                
                calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
                set_running_statistics(model, calib_dataloader, 10)

            # 获取teacher特征
            with torch.no_grad():
                # 切换到最大网络获取teacher特征
                ofa_network.set_active_subnet(**max_net_config)
                set_running_statistics(model, calib_dataloader, 10)
                
                teacher_fpn_features = []
                for img in images:
                    feat = model.backbone(img.unsqueeze(0))
                    teacher_fpn_features.append(feat)
                
                # 切回当前子网络配置
                ofa_network.set_active_subnet(**current_config)
                set_running_statistics(model, calib_dataloader, 10)

            # 获取student特征并计算检测损失
            student_fpn_features = []
            for img in images:
                feat = model.backbone(img.unsqueeze(0))
                student_fpn_features.append(feat)

            # 计算检测损失
            det_loss_dict = model(images, targets)
            det_loss = sum(loss for loss in det_loss_dict.values())
            
            # 计算蒸馏损失
            batch_distill_loss = 0
            for t_fpn, s_fpn in zip(teacher_fpn_features, student_fpn_features):
                batch_distill_loss += fpn_distill_loss(t_fpn, s_fpn)
            distillation_loss = batch_distill_loss / len(images)
            
            # 总损失
            total_loss = det_loss + distill_alpha * distillation_loss

            if i % subnet_sample_interval == 0:
                logger.info(f"Det loss: {det_loss.item():.4f}, Distill loss: {distillation_loss.item():.4f}")

            optimizer_backbone.zero_grad()
            optimizer_head.zero_grad()
            total_loss.backward()
            optimizer_backbone.step()
            optimizer_head.step()

            loss_sum += total_loss.item()

            if i > 0 and i % subnet_sample_interval == 0:
                logger.info(f"Iteration #{i} loss: {loss_sum / subnet_sample_interval}")
                torch.cuda.empty_cache()
                loss_sum = 0
            i += 1
        
        logger.info(f"Epoch {epoch+1} finished.")
        scheduler_backbone.step()
        scheduler_head.step()
        logger.info(f"Current learning rates - backbone: {scheduler_backbone.get_last_lr()}, head: {scheduler_head.get_last_lr()}")
        
        # 评估所有预定义配置的性能
        for config in subnet_configs:
            ofa_network.set_active_subnet(**config)
            calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
            set_running_statistics(model, calib_dataloader, 10)
            acc = eval_accuracy(model, None, 100, device, show_progress=False)['AP@0.5:0.95']
            logger.info(f"Subnet config: {config}, accuracy: {acc}")

        # 恢复到当前训练的配置
        if current_config:
            ofa_network.set_active_subnet(**current_config)
            calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
            set_running_statistics(model, calib_dataloader, 10)

        torch.save(model, save_path)

    logger.info("Training complete.")
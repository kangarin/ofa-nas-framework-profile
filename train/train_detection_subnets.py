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

def train_selected_subnets(model, subnet_configs, num_epochs, save_path,
                          batch_size=1,
                          backbone_learning_rate=1e-4, head_learning_rate=1e-4,
                          min_backbone_lr=1e-4, min_head_lr=1e-4,
                          subnet_sample_interval=5):
    """
    Train detection network using selected subnet configurations.
    
    Args:
        model: Model to train
        subnet_configs: List of subnet configurations to sample from
        num_epochs: Number of training epochs
        save_path: Path to save trained model
        batch_size: Batch size for training
        backbone_learning_rate: Initial learning rate for backbone
        head_learning_rate: Initial learning rate for detection head
        min_backbone_lr: Minimum learning rate for backbone
        min_head_lr: Minimum learning rate for detection head
        subnet_sample_interval: Interval for switching subnet configurations
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
    
    # 创建优化器
    optimizer_backbone = torch.optim.Adam(params_backbone, lr=backbone_learning_rate)
    optimizer_head = torch.optim.Adam(params_head, lr=head_learning_rate)
    
    # 创建学习率调度器
    scheduler_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_backbone,
        T_max=num_epochs,
        eta_min=min_backbone_lr,
        verbose=True
    )
    
    scheduler_head = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_head,
        T_max=num_epochs,
        eta_min=min_head_lr,
        verbose=True
    )

    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Start training, using device: {device}")
    model.to(device)

    # 准备校准数据集
    calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_list))
    calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
    
    # 准备训练数据
    train_dataloader = get_dataloader(get_train_dataset(), batch_size)
    
    # 获取OFA网络
    ofa_network = model.backbone.body
    
    # 记录当前使用的配置
    current_config = None
    
    # 开始训练
    for epoch in range(num_epochs):
        model.train()
        i = 0
        loss_sum = 0
        
        for data in train_dataloader:
            if not data:  # 跳过空数据
                continue
                
            # 临时用于本地测试，正式训练应该删除
            if i > 1000:
                break
                
            # 准备数据
            images, targets = data
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 每subnet_sample_interval次迭代切换一次网络配置
            if i % subnet_sample_interval == 0:
                # 从预定义的配置列表中随机选择一个
                current_config = random.choice(subnet_configs)
                ofa_network.set_active_subnet(**current_config)
                logger.info(f"Switched to subnet config: {current_config}")
                
                # 重新校准BN统计量
                calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
                set_running_statistics(model, calib_dataloader, 10)

            # 前向传播和损失计算
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 反向传播和优化
            optimizer_backbone.zero_grad()
            optimizer_head.zero_grad()
            losses.backward()
            optimizer_backbone.step()
            optimizer_head.step()

            loss_sum += losses.item()

            # 打印训练信息
            if i > 0 and i % subnet_sample_interval == 0:
                logger.info(f"Iteration #{i} loss: {loss_sum / subnet_sample_interval}")
                torch.cuda.empty_cache()
                loss_sum = 0
            i += 1
        
        # 每个epoch结束
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

        # 保存模型
        torch.save(model, save_path)

    logger.info("Training complete.")
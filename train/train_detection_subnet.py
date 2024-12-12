from utils.bn_calibration import set_running_statistics
from datasets.calib_dataset import get_calib_dataset, create_fixed_size_dataloader
from datasets.coco_dataset import get_train_dataset, get_dataloader
import torch
from datasets.common_transform import common_transform_list
from torchvision import transforms
from utils.logger import setup_logger

logger = setup_logger('train')

def train_subnet(model, subnet_config, num_epochs, save_path,
                batch_size=1,
                backbone_learning_rate=1e-3, 
                head_learning_rate=1e-2,
                min_backbone_lr=1e-5, 
                min_head_lr=1e-4):
    """
    Train a specific subnet configuration of the detection network.
    
    Args:
        model: Model to train
        subnet_config: Configuration dictionary for the subnet
        num_epochs: Number of training epochs
        save_path: Path to save trained model
        batch_size: Batch size for training
        backbone_learning_rate: Initial learning rate for backbone
        head_learning_rate: Initial learning rate for detection head
        min_backbone_lr: Minimum learning rate for backbone
        min_head_lr: Minimum learning rate for detection head
    """
    # Set up optimizers
    params_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if hasattr(model, 'head'):
        params_head = [p for p in model.head.parameters() if p.requires_grad]
    elif hasattr(model, 'roi_heads'):
        params_head = [p for p in model.roi_heads.parameters() if p.requires_grad]

    params_backbone = [{'params': params_backbone, 'lr': backbone_learning_rate}]
    params_head = [{'params': params_head, 'lr': head_learning_rate}]
    
    optimizer_backbone = torch.optim.SGD(params_backbone, 
                                       lr=backbone_learning_rate, 
                                       momentum=0.9, 
                                       weight_decay=1e-4)
    optimizer_head = torch.optim.SGD(params_head, 
                                    lr=head_learning_rate, 
                                    momentum=0.9, 
                                    weight_decay=1e-4)
    
    # Learning rate schedulers
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

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Start training, using device: {device}")
    model.to(device)

    # Set subnet configuration
    ofa_network = model.backbone.body
    ofa_network.set_active_subnet(**subnet_config)
    logger.info(f"Training with subnet config: {subnet_config}")

    # Prepare calibration dataset
    calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_list))
    calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
    set_running_statistics(model, calib_dataloader, 10)

    # Prepare training data
    train_dataloader = get_dataloader(get_train_dataset(), batch_size)
    
    # Start training
    for epoch in range(num_epochs):
        model.train()
        i = 0
        loss_sum = 0
        
        for data in train_dataloader:
            if not data:  # Skip empty data
                continue
                
            # For local testing only, remove in production
            if i > 1000:
                break
                
            # Prepare data
            images, targets = data
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass and loss calculation
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimization
            optimizer_backbone.zero_grad()
            optimizer_head.zero_grad()
            losses.backward()
            optimizer_backbone.step()
            optimizer_head.step()

            loss_sum += losses.item()

            # Print training information
            if i > 0 and i % 100 == 0:  # Print every 100 iterations
                logger.info(f"Epoch {epoch+1}, Iteration #{i}, avg loss: {loss_sum / 100}")
                torch.cuda.empty_cache()
                loss_sum = 0
            i += 1
        
        # End of epoch
        logger.info(f"Epoch {epoch+1} finished.")
        scheduler_backbone.step()
        scheduler_head.step()
        logger.info(f"Current learning rates - backbone: {scheduler_backbone.get_last_lr()}, head: {scheduler_head.get_last_lr()}")
        
        # Save model
        torch.save(model, save_path)

    logger.info("Training complete.")

from utils.bn_calibration import set_running_statistics
from datasets.calib_dataset import get_calib_dataset, create_fixed_size_dataloader
from datasets.coco_dataset import get_train_dataset, get_dataloader
import torch

from utils.logger import setup_logger
logger = setup_logger('train')

def train(model, num_epochs, save_path, batch_size = 1, backbone_learning_rate = 1e-4, head_learning_rate = 1e-5, subnet_sample_interval = 20):

    # 设置优化器
    params_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    params_head = [p for p in model.head.parameters() if p.requires_grad]

    params = [{'params': params_backbone, 'lr': backbone_learning_rate}, {'params': params_head, 'lr': head_learning_rate}]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Start training, using device: {device}")
    model.to(device)
    # 训练模型
    calib_dataset = get_calib_dataset()
    calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
    set_running_statistics(model, calib_dataloader, 10)

    train_dataloader = get_dataloader(get_train_dataset(), batch_size)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        loss_sum = 0
        for data in train_dataloader:
            if not data:  # 如果data为空
                continue  # 跳过当前迭代，继续下一个迭代
            # coco数据集太大，本地微调只用了一部分，正式训练应该删除
            if i > 5000:
                break        
            images, targets = data
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            ofa_network = model.backbone.body
            # 第一轮，直接用max subnet来确保动态的连接层能学到比较好的初始值
            if epoch == -1:
                ofa_network.set_max_net()
            # 否则，用sample出来的子网络
            else:
                # model的backbone.body随机sample一个子网络（subnet_sample_interval轮sample一次，该值太小可能训练不稳定，太大可能采样子网络不充分）
                if i % subnet_sample_interval == 0:
                    subnet_config = ofa_network.sample_active_subnet()
                    ofa_network.set_active_subnet(**subnet_config)
                    # 这里每次都重新创建data_loader，增加calib的随机性，从而提高泛化能力
                    calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10)
                    set_running_statistics(model, calib_dataloader, 10)

            # 计算损失
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 优化器梯度清零
            optimizer.zero_grad()
            # 反向传播
            losses.backward()
            # 更新参数
            optimizer.step()

            loss_sum += losses.item()

            # 每subnet_sample_interval次打印损失
            if i > 0 and i % subnet_sample_interval == 0:
                logger.info(f"Iteration #{i} loss: {loss_sum / subnet_sample_interval}")
                # 清除CUDA缓存
                torch.cuda.empty_cache()
                loss_sum = 0
            i += 1
        
        logger.info(f"Epoch {epoch+1} finished.")
        torch.save(model, save_path)

    logger.info("Training complete.")
from torchvision import transforms
import torch

common_transform_list = [
        transforms.ToTensor(),
]

# TODO: 需要看一下分类网络对normalization的要求
common_transform_with_normalization_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

def collate_fn_pad(batch):
    '''
    带有gt数据集的collate_fn
    '''
    # 找到这个batch中最大的高度和宽度
    max_h = max([img.shape[1] for img, _ in batch])
    max_w = max([img.shape[2] for img, _ in batch])
    
    # 填充每张图片到最大尺寸
    images = []
    for img, target in batch:
        # 假设img是 [C, H, W]
        h, w = img.shape[1:]
        # 创建填充后的张量
        pad_img = torch.zeros((3, max_h, max_w))
        # 复制原图数据
        pad_img[:, :h, :w] = img
        images.append(pad_img)
    
    # 堆叠图片和标签
    images = torch.stack(images)
    targets = torch.tensor([t for _, t in batch])
    
    return images, targets

def collate_fn_pad_calibration(batch):
    '''
    用于标定数据集的collate_fn，不需要标签
    '''
    # 找到这个batch中最大的高度和宽度
    max_h = max([img.shape[1] for img in batch])
    max_w = max([img.shape[2] for img in batch])
    
    # 填充每张图片到最大尺寸
    images = []
    for img in batch:
        # 假设img是 [C, H, W]
        h, w = img.shape[1:]
        # 创建填充后的张量
        pad_img = torch.zeros((3, max_h, max_w))
        # 复制原图数据
        pad_img[:, :h, :w] = img
        images.append(pad_img)
    
    # 堆叠图片和标签
    images = torch.stack(images)
    
    return images

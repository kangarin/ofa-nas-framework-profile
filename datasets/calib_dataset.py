# 专门为calibration写的dataset，因为不需要gt，只要一个图片文件夹用于校准
# 注意transform可能要修改，应当与训练保持一样
# 同时也要注意推理的时候也需要做transform，比如归一化，之前没注意，可能影响精度

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Callable, List, Tuple
from config import Config
from datasets.common_transform import common_transform_list, collate_fn_pad_calibration
import random
from torch.utils.data import Subset

from utils.logger import setup_logger
logger = setup_logger('dataset')

class SimpleImageDataset(Dataset):
    """
    A simple dataset class that only loads images without ground truth
    Specifically designed for batch norm related measurements
    """
    def __init__(
        self,
        img_dir: str,
        transform: Optional[Callable] = None,
        valid_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        self.img_dir = img_dir
        self.transform = transform
        
        # Get all valid image files
        self.img_files = []
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(valid_extensions):
                self.img_files.append(os.path.join(img_dir, filename))
                
        if not self.img_files:
            raise RuntimeError(f"Found 0 images in {img_dir}")
            
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load image
        img_path = self.img_files[idx]
        with Image.open(img_path).convert('RGB') as img:
            if self.transform:
                img = self.transform(img)
            return img

def get_calib_dataset(custom_transform: Optional[Callable], 
                      img_dir: str = Config.CALIB_DIR) -> SimpleImageDataset:
    '''
    !!!!!!!!! 必须要注意，这里的transform必须要和训练时的transform保持一致，否则会影响精度 !!!!!!!!!
    比如，分类模型要normalize，检测模型不要normalize
    '''
    # 创建完整数据集
    dataset = SimpleImageDataset(
        img_dir=img_dir,
        transform=custom_transform
    )
    return dataset

def create_dataloader(
    dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    custom_transform: Optional[Callable] = None
) -> DataLoader:

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle to get random batches
        num_workers=num_workers,
        collate_fn=collate_fn_pad_calibration
    )

    return dataloader

def create_fixed_size_dataloader(
    dataset: SimpleImageDataset,
    num_images: int = 10,  # 想要的图片数量
    batch_size: int = 1, 
    num_workers: int = 0,
    custom_transform: Optional[Callable] = None,
    seed: int = 42  # 可选的随机种子，保证每次采样结果一致
) -> DataLoader:
    
    # 确保请求的图片数量不超过数据集大小
    num_images = min(num_images, len(dataset))
    
    # 设置随机种子以保证可重复性
    # random.seed(seed)
    
    # 随机采样指定数量的索引
    indices = random.sample(range(len(dataset)), num_images)

    # 打印采样的索引
    logger.info(f"Calib dataloader sampled indices: {indices}")
    
    # 创建子数据集
    subset_dataset = Subset(dataset, indices)
    
    # 创建数据加载器
    dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_pad_calibration
    )

    return dataloader

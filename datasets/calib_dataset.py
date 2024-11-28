# 专门为calibration写的dataset，因为不需要gt，只要一个图片文件夹用于校准
# 注意transform可能要修改，应当与训练保持一样
# 同时也要注意推理的时候也需要做transform，比如归一化，之前没注意，可能影响精度
# TODO

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Callable, List, Tuple
from config import Config
from datasets.common_transform import common_transform_list

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

# Example usage
def create_dataloader(
    img_dir: str,
    batch_size: int = 1,
    num_workers: int = 0,
    custom_transform: Optional[Callable] = None
) -> DataLoader:
    """
    Create a DataLoader for the image dataset
    
    Args:
        img_dir: Directory containing images
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
        custom_transform: Optional custom transform pipeline
        
    Returns:
        DataLoader instance
    """
    dataset = SimpleImageDataset(
        img_dir=img_dir,
        transform=custom_transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle to get random batches
        num_workers=num_workers,
    )

    return dataloader

def get_calib_dataloader():
    data_loader = create_dataloader(Config.CALIB_DIR, custom_transform=transforms.Compose(common_transform_list))
    return data_loader

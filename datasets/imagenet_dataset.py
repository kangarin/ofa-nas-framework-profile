from config import Config
imagenet_dir = Config.IMAGENET_DIR
from datasets.common_transform import common_transform_with_normalization_list, collate_fn_pad
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
import torch

def get_train_dataset():
    train_dataset = ImageNet(
        root=imagenet_dir,
        split='train',
        transform=transforms.Compose(common_transform_with_normalization_list)
    )
    return train_dataset

def get_test_dataset():
    test_dataset = ImageNet(
        root=imagenet_dir,
        split='val',
        transform=transforms.Compose(common_transform_with_normalization_list)
    )
    return test_dataset

def get_dataloader(dataset, batch_size):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_pad
    )
    return dataloader

def get_classes():
    return get_test_dataset().classes

def get_class_to_idx():
    return get_test_dataset().class_to_idx
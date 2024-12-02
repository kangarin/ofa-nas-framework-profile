from config import Config
from datasets.common_transform import common_transform_list
coco_datadir = Config.COCO_DIR

import torch
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import os

def transform(image, target):

    for t in common_transform_list:
        image = t(image)
     
    # 转换bbox格式
    boxes = []
    labels = []
    if len(target) > 0:
        for obj in target:
            # 如果width或height为0，则跳过
            if obj['bbox'][2] == 0 or obj['bbox'][3] == 0:
                continue
            x_min, y_min, width, height = obj['bbox']
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(obj['category_id'])
    
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    
    # 获取image_id
    image_id = torch.tensor([target[0]['image_id']]) if len(target) > 0 else torch.tensor([-1])
    
    target = {'boxes': boxes, 'labels': labels, 'image_id': image_id}
    
    return image, target

# 过滤掉没有标注对象的样本
def collate_fn(batch):
    batch = list(filter(lambda x: len(x[1]['boxes']) > 0, batch))
    return tuple(zip(*batch))

def get_train_dataset():
    train_dataset = CocoDetection(
        root=os.path.join(coco_datadir, 'train2017'),
        annFile=os.path.join(coco_datadir, 'annotations', 'instances_train2017.json'),
        transforms=transform
    )
    return train_dataset

def get_test_dataset():
    test_dataset = CocoDetection(
        root=os.path.join(coco_datadir, 'val2017'),
        annFile=os.path.join(coco_datadir, 'annotations', 'instances_val2017.json'),
        transforms=transform
    )
    return test_dataset

def get_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size, shuffle = True, collate_fn = collate_fn)

def get_labels():
    return coco_labels

# COCO标签映射
coco_labels = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
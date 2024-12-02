import random
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import torch
from tqdm import tqdm
import config
import os
from PIL import Image
from datasets.common_transform import common_transform_list
import torchvision.transforms.functional as F

def calculate_new_size(original_size, max_size):
    """
    计算等比例缩放后的尺寸
    Args:
        original_size: 原始尺寸 (width, height)
        max_size: 较长边的目标长度
    Returns:
        new_width, new_height: 缩放后的宽度和高度
    """
    width, height = original_size
    # 确定较长边
    if width >= height:
        # 如果宽度更长
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        # 如果高度更长
        new_height = max_size
        new_width = int(width * max_size / height)
    return new_width, new_height

def eval_accuracy(model, input_size, image_nums, device, show_progress=True):
    """
    评估模型准确率，支持图像等比例缩放
    Args:
        model: 要评估的模型
        device: 计算设备
        image_nums: 要评估的图像数量
        input_size: 较长边的目标长度，如果为 None，则保持原始尺寸
        show_progress: 是否显示进度条
    """
    # 加载注释文件
    ann_file = config.Config.COCO_ANN_VAL_FILE
    coco = COCO(ann_file)
    model = model.to(device)
    model.eval()

    # 加载图像文件夹路径
    img_dir = os.path.join(config.Config.COCO_DIR, 'val2017')

    # 随机选择 n 张图像
    img_ids = list(coco.imgs.keys())
    selected_img_ids = random.sample(img_ids, image_nums)

    # 初始化模型
    model.eval()

    # 预测结果列表
    results = []

    # 遍历选中的图像
    for img_id in tqdm(selected_img_ids, desc="Evaluating", disable=not show_progress):
        # 加载图像
        img_info = coco.loadImgs(img_id)[0]
        img_path = f"{img_dir}/{img_info['file_name']}"
        image = Image.open(img_path).convert("RGB")
        
        # 记录原始尺寸
        original_size = image.size  # (width, height)
        
        # 如果指定了输入尺寸，进行等比例缩放
        if input_size is not None:
            # 计算新的尺寸
            new_width, new_height = calculate_new_size(original_size, input_size)
            
            # 计算缩放比例
            scale_w = new_width / original_size[0]
            scale_h = new_height / original_size[1]
            
            # 调整图像大小
            image = F.resize(image, (new_height, new_width))

        # 应用其他转换
        for t in common_transform_list:
            image = t(image)
            
        # 添加 batch 维度
        image = image.unsqueeze(0).to(device)

        # 获取预测结果
        with torch.no_grad():
            predictions = model(image)[0]

        # 处理预测结果
        for idx in range(len(predictions['boxes'])):
            box = predictions['boxes'][idx].cpu().numpy().tolist()
            score = predictions['scores'][idx].cpu().item()
            label = predictions['labels'][idx].cpu().item()
            
            # 如果进行了缩放，将检测框转换回原始尺寸
            if input_size is not None:
                # 反向缩放检测框坐标
                x1 = box[0] / scale_w
                y1 = box[1] / scale_h
                x2 = box[2] / scale_w
                y2 = box[3] / scale_h
                
                # 计算宽度和高度（COCO格式要求）
                width = x2 - x1
                height = y2 - y1
                
                bbox = [x1, y1, width, height]
            else:
                # 如果没有缩放，直接转换为COCO格式（x,y,width,height）
                bbox = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            
            results.append({
                "image_id": img_id,
                "category_id": label,
                "bbox": bbox,
                "score": score
            })

    # 保存预测结果为 JSON 文件
    result_file = 'results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f)

    # 创建过滤后的注释文件
    filtered_annotations = {
        "images": [img for img in coco.dataset["images"] if img["id"] in selected_img_ids],
        "annotations": [ann for ann in coco.dataset["annotations"] if ann["image_id"] in selected_img_ids],
        "categories": coco.dataset["categories"]
    }

    # 保存过滤后的注释文件
    filtered_ann_file = 'filtered_annotations.json'
    with open(filtered_ann_file, 'w') as f:
        json.dump(filtered_annotations, f)

    # 使用过滤后的注释文件进行评估
    coco_filtered = COCO(filtered_ann_file)
    coco_dt = coco_filtered.loadRes(result_file)
    coco_eval = COCOeval(coco_filtered, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 返回评估指标
    metrics = {
        'AP@0.5:0.95': coco_eval.stats[0],
        'AP@0.5': coco_eval.stats[1],
        'AP@0.75': coco_eval.stats[2],
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5],
        'AR@1': coco_eval.stats[6],
        'AR@10': coco_eval.stats[7],
        'AR@100': coco_eval.stats[8],
    }

    # 删除临时文件
    os.remove(result_file)
    os.remove(filtered_ann_file)

    return metrics
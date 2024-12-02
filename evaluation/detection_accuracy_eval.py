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
def eval_accuracy(model, device, image_nums, show_progress=True):
    # 加载注释文件
    ann_file = config.Config.COCO_ANN_VAL_FILE
    coco = COCO(ann_file)

    # 加载图像文件夹路径
    img_dir = os.path.join(config.Config.COCO_DIR, 'val2017')

    # 随机选择 n 张图像
    img_ids = list(coco.imgs.keys())
    # random.seed(42)  # 设置随机种子以保证结果可复现
    selected_img_ids = random.sample(img_ids, image_nums)

    import torch

    # 初始化模型
    model.eval()

    # 预测结果列表
    results = []

    # 遍历COCO验证集中的所有图像
    for img_id in tqdm(selected_img_ids, desc="Evaluating"):
        # 加载图像
        img_info = coco.loadImgs(img_id)[0]
        img_path = f"{img_dir}/{img_info['file_name']}"
        image = Image.open(img_path).convert("RGB")
        # 将 Image 对象转换为张量
        for t in common_transform_list:
            image = t(image)
        # 添加 batch 维度
        image = image.unsqueeze(0)
        model.cuda()
        image = image.cuda()

        # 获取预测结果
        with torch.no_grad():
            predictions = model(image)[0]

        # 处理预测结果
        for idx in range(len(predictions['boxes'])):
            box = predictions['boxes'][idx].cpu().numpy().tolist()
            score = predictions['scores'][idx].cpu().item()
            label = predictions['labels'][idx].cpu().item()
            results.append({
                "image_id": img_id,
                "category_id": label,
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                "score": score
            })

    # 保存预测结果为 JSON 文件
    result_file = 'results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f)

    # 创建一个新的注释文件，仅包含随机选择的100张图像及其相关注释
    filtered_annotations = {
        "images": [img for img in coco.dataset["images"] if img["id"] in selected_img_ids],
        "annotations": [ann for ann in coco.dataset["annotations"] if ann["image_id"] in selected_img_ids],
        "categories": coco.dataset["categories"]
    }

    # 保存过滤后的注释文件
    filtered_ann_file = 'filtered_annotations.json'
    with open(filtered_ann_file, 'w') as f:
        json.dump(filtered_annotations, f)

    # 使用新的注释文件进行评估
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
    return metrics

import torch

from datasets.calib_dataset import create_fixed_size_dataloader, get_calib_dataset
from utils.bn_calibration import set_running_statistics

def test_model_api():
    from models.detection.ofa_resnet50_fcos import get_ofa_resnet50_fcos_model, load_pretrained_fcos, set_training_params
    model = get_ofa_resnet50_fcos_model()
    load_pretrained_fcos(model)
    set_training_params(model)

def train_fcos_mbv3_w12():
    from train.train_detection_networks import train
    from models.detection.ofa_mbv3_w12_fcos import get_ofa_mbv3_w12_fcos_model, load_pretrained_fcos, set_training_params
    model = get_ofa_mbv3_w12_fcos_model()
    load_pretrained_fcos(model)
    set_training_params(model)
    train(model, 5, 'ofa_mbv3_w12_fcos.pth', batch_size=2)
    model = torch.load('ofa_mbv3_w12_fcos.pth')

def train_fcos_resnet50():
    from train.train_detection_networks import train
    from models.detection.ofa_resnet50_fcos import get_ofa_resnet50_fcos_model, load_pretrained_fcos, set_training_params
    model = get_ofa_resnet50_fcos_model()
    load_pretrained_fcos(model)
    set_training_params(model)
    train(model, 5, 'ofa_resnet50_fcos.pth', batch_size=2)
    model = torch.load('ofa_resnet50_fcos.pth')    

def train_fasterrcnn_mbv3_w12():
    from train.train_detection_networks import train
    from models.detection.ofa_mbv3_w12_fasterrcnn import get_ofa_mbv3_w12_fasterrcnn_model, load_pretrained_fasterrcnn, set_training_params
    model = get_ofa_mbv3_w12_fasterrcnn_model()
    load_pretrained_fasterrcnn(model)
    set_training_params(model)
    train(model, 5, 'ofa_mbv3_w12_fasterrcnn.pth', batch_size=2)
    model = torch.load('ofa_mbv3_w12_fasterrcnn.pth')

def train_fasterrcnn_resnet50():
    from train.train_detection_networks import train
    from models.detection.ofa_resnet50_fasterrcnn import get_ofa_resnet50_fasterrcnn_model, load_pretrained_fasterrcnn, set_training_params
    model = get_ofa_resnet50_fasterrcnn_model()
    load_pretrained_fasterrcnn(model)
    set_training_params(model)
    train(model, 5, 'ofa_resnet50_fasterrcnn.pth', batch_size=2)
    model = torch.load('ofa_resnet50_fasterrcnn.pth')

def test_det_api():
    import torch
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    model = torch.load('ofa_mbv3_w12_fcos.pth')
    # model = torch.load('ofa_resnet50_fcos.pth')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    image_path1 = "D:\Projects\python\smart-dnn-framework\my_experiments\ILSVRC2012_val_00000384.JPEG"
    image_path2 = "D:\\Projects\\python\\once-for-all\\00300588922_1f374172.jpg"

    from inference.detection_inference import DetectionInference
    detection_inference = DetectionInference(model, device)

    for i in range(10):
        subnet_config = model.backbone.body.sample_active_subnet()
        detection_inference.set_active_subnet(**subnet_config)
        print(subnet_config)

        # 保存原始图像用于显示
        pil_img1 = Image.open(image_path1).convert("RGB")
        pil_img2 = Image.open(image_path2).convert("RGB")
        original_images = [pil_img1, pil_img2]

        # 转换图像为tensor并进行标准化
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        img1 = transform(pil_img1)
        img2 = transform(pil_img2)

        from utils.common import resize_images
        img = resize_images([img1, img2])

        # 推理
        batch_boxes, batch_labels, batch_scores = detection_inference.detect(img)

        # 显示原始图像和检测结果
        for orig_img, boxes, labels, scores in zip(original_images, batch_boxes, batch_labels, batch_scores):
            plt.figure(figsize=(10, 8))
            # 显示原始PIL图像
            plt.imshow(orig_img)
            ax = plt.gca()
            
            # 确保边界框坐标与原始图像尺寸匹配
            img_width, img_height = orig_img.size
            for box, label, score in zip(boxes, labels, scores):
                # 绘制边界框
                x_min, y_min, x_max, y_max = box
                # 确保坐标不超出图像边界
                x_min = max(0, min(x_min, img_width))
                x_max = max(0, min(x_max, img_width))
                y_min = max(0, min(y_min, img_height))
                y_max = max(0, min(y_max, img_height))
                
                box_width = x_max - x_min
                box_height = y_max - y_min
                
                rect = plt.Rectangle((x_min, y_min), box_width, box_height,
                                   fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                
                # 添加类别和置信度标签
                from datasets.coco_dataset import coco_labels
                class_name = coco_labels[label]
                plt.text(x_min, y_min - 5, f'{class_name} {score:.2f}',
                        color='red', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0))
            
            plt.axis('off')
            plt.show()

def test_calib_bs():
    calib_dataset = get_calib_dataset() 
    calib_dataloader = create_fixed_size_dataloader(calib_dataset, 10, batch_size=2)
    from models.detection.ofa_mbv3_w12_fcos import get_ofa_mbv3_w12_fcos_model, load_pretrained_fcos, set_training_params
    model = get_ofa_mbv3_w12_fcos_model()
    set_running_statistics(model, calib_dataloader)

def test_fpn_with_other_det():
    from models.fpn.ofa_supernet_mbv3_w12_fcos_fpn import Mbv3W12FcosFpn
    from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
    fpn = Mbv3W12FcosFpn(get_ofa_supernet_mbv3_w12())
    from torchvision.models.detection import FasterRCNN
    model = FasterRCNN(backbone = fpn, num_classes = 91)
    print(model)
    import torch
    # test forward and backward
    for i in range(10):
        subnet_config = model.backbone.body.sample_active_subnet()
        model.backbone.body.set_active_subnet(**subnet_config)
        print(subnet_config)
        images = torch.randn(2, 3, 224, 224)
        targets = [{'boxes': torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150]]), 'labels': torch.tensor([1, 2])},
                   {'boxes': torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150]]), 'labels': torch.tensor([1, 2])}]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        print(losses)

if __name__ == '__main__':
    # train_fcos_mbv3_w12()
    # train_fcos_resnet50()
    # train_fasterrcnn_mbv3_w12()
    train_fasterrcnn_resnet50()
    # test_calib_bs()
    # test_det_api()
    # test_model_api()
    # test_fpn_with_other_det()

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
    from models.backbone.ofa_supernet import get_max_net_config, get_min_net_config
    max_net_config = get_max_net_config(ofa_supernet_name='ofa_supernet_mbv3_w12')
    min_net_config = get_min_net_config(ofa_supernet_name='ofa_supernet_mbv3_w12')
    # load pre-trained weights if exists
    import os
    if os.path.exists('ofa_mbv3_w12_fcos.pth'):
        model = torch.load('ofa_mbv3_w12_fcos.pth')
    load_pretrained_fcos(model)
    set_training_params(model, is_backbone_body_need_training=False)
    train(model, 100, 'ofa_mbv3_w12_fcos.pth', max_net_config, min_net_config,
          batch_size=4, 
          backbone_learning_rate=5e-3, head_learning_rate=1e-2,
          min_backbone_lr=5e-5, min_head_lr=1e-4)
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
    from models.backbone.ofa_supernet import get_max_net_config, get_min_net_config
    max_net_config = get_max_net_config(ofa_supernet_name='ofa_supernet_mbv3_w12')
    min_net_config = get_min_net_config(ofa_supernet_name='ofa_supernet_mbv3_w12')
    model = get_ofa_mbv3_w12_fasterrcnn_model()
    load_pretrained_fasterrcnn(model)
    set_training_params(model, is_backbone_body_need_training=False)
    train(model, 5, 'ofa_mbv3_w12_fasterrcnn.pth', max_net_config, min_net_config, batch_size=2)
    model = torch.load('ofa_mbv3_w12_fasterrcnn.pth')

def train_fasterrcnn_resnet50():
    from train.train_detection_networks import train
    from models.detection.ofa_resnet50_fasterrcnn import get_ofa_resnet50_fasterrcnn_model, load_pretrained_fasterrcnn, set_training_params
    model = get_ofa_resnet50_fasterrcnn_model()
    load_pretrained_fasterrcnn(model)
    set_training_params(model)
    train(model, 5, 'ofa_resnet50_fasterrcnn.pth', batch_size=2)
    model = torch.load('ofa_resnet50_fasterrcnn.pth')

def test_classification_api():
    from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
    model = get_ofa_supernet_mbv3_w12()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    image_path1 = "C:\\Users\\wydai\\Downloads\\111121084738-traffic-jam.jpg"
    image_path2 = "D:\\Projects\\python\\once-for-all\\00300588922_1f374172.jpg" 

    from inference.classification_inference import ClassificationInference
    from datasets.imagenet_dataset import get_test_dataset, get_dataloader
    dataset = get_test_dataset()
    dataloader = get_dataloader(dataset, 4)
    infer = ClassificationInference(model, device)
    from PIL import Image
    pil_img1 = Image.open(image_path1).convert("RGB")
    pil_img2 = Image.open(image_path2).convert("RGB")
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img1 = transform(pil_img1)
    img2 = transform(pil_img2)
    from utils.common import resize_images
    img = resize_images([img1, img2])
    result = infer.classify(img)
    max_score, max_label = torch.max(result, 1)
    from datasets.imagenet_dataset import get_classes
    print(get_classes()[max_label[0].item()])
    print(get_classes()[max_label[1].item()])
    print(max_score, max_label)   

def test_det_api():
    import torch
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # model = torch.load('ofa_mbv3_w12_fcos.pth')
    model = torch.load('ofa_mbv3_w12_fcos_subnet.pth')
    from models.backbone.ofa_supernet import get_max_net_config, get_min_net_config
    max_net_config = get_max_net_config(ofa_supernet_name='ofa_supernet_mbv3_w12')
    min_net_config = get_min_net_config(ofa_supernet_name='ofa_supernet_mbv3_w12')
    # model = torch.load('ofa_resnet50_fcos.pth')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # image_path1 = "C:\\Users\\wydai\\Downloads\\ladwa-40.png"
    image_path1 = "D:\\Projects\\python\\once-for-all\\00300588922_1f374172.jpg"
    image_path2 = "D:\\Projects\\coco2017\\val2017\\000000142238.jpg"

    from inference.detection_inference import DetectionInference
    detection_inference = DetectionInference(model, device)

    for i in range(10):
        subnet_config = model.backbone.body.sample_active_subnet()
        # detection_inference.set_active_subnet(**subnet_config)
        # detection_inference.model.backbone.body.set_max_net()
        detection_inference.set_active_subnet(**min_net_config)
        set_running_statistics(detection_inference.model, detection_inference.calib_dataloader, 10)
        print(subnet_config)

        # from evaluation.detection_accuracy_eval import eval_accuracy
        # eval_accuracy(model, 640, 100, device)

        # 保存原始图像用于显示
        pil_img1 = Image.open(image_path1).convert("RGB")
        pil_img2 = Image.open(image_path2).convert("RGB")
        original_images = [pil_img1, pil_img2]

        # 转换图像为tensor
        transform = T.Compose([
            T.ToTensor(),
        ])
        
        img1 = transform(pil_img1)
        img2 = transform(pil_img2)

        from utils.common import resize_images
        img = resize_images([img1, img2])

        # 推理
        batch_boxes, batch_labels, batch_scores = detection_inference.detect(img, 0.45)

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
    from datasets.common_transform import common_transform_list
    from torchvision import transforms
    calib_dataset = get_calib_dataset(custom_transform=transforms.Compose(common_transform_list))
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

def some_test():
    '''
    test imagenet dataset api
    '''
    from inference.classification_inference import ClassificationInference
    from datasets.imagenet_dataset import get_test_dataset, get_dataloader
    dataset = get_test_dataset()
    dataloader = get_dataloader(dataset, 4)
    from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
    model = get_ofa_supernet_mbv3_w12()
    # model.set_max_net()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    infer = ClassificationInference(model, device)
    image_path1 = "D:\Projects\python\smart-dnn-framework\my_experiments\ILSVRC2012_val_00000384.JPEG"
    image_path2 = "D:\\Projects\\python\\once-for-all\\00300588922_1f374172.jpg"
    from PIL import Image
    pil_img1 = Image.open(image_path1).convert("RGB")
    pil_img2 = Image.open(image_path2).convert("RGB")
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img1 = transform(pil_img1)
    img2 = transform(pil_img2)
    from utils.common import resize_images
    img = resize_images([img1, img2])
    result = infer.classify(img)
    max_score, max_label = torch.max(result, 1)
    print(max_score, max_label)
    from datasets.imagenet_dataset import get_classes
    print(get_classes()[max_label[0].item()])
    print(get_classes()[max_label[1].item()])
    from evaluation.classification_accuracy_eval import eval_accuracy
    input_size = 224
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    result = eval_accuracy(model, input_size, dataloader, 100, device, (1, 5))
    print(f'Accuracy: {result}')

def some_test1():
    '''
    test coco dataset api
    '''
    from datasets.coco_dataset import get_test_dataset, get_dataloader
    dataset = get_test_dataset()
    dataloader = get_dataloader(dataset, 4)
    from torchvision.models.detection.fcos import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
    model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    for image, _ in dataloader:
        result = model(image)
        print(result)

def some_test2():
    '''
    test calib dataset api
    '''
    from datasets.calib_dataset import get_calib_dataset, create_fixed_size_dataloader
    dataset = get_calib_dataset()
    dataloader = create_fixed_size_dataloader(dataset, 10, 2)
    from models.detection.ofa_resnet50_fcos import get_ofa_resnet50_fcos_model
    model = get_ofa_resnet50_fcos_model()
    set_running_statistics(model, dataloader, 10)

def some_test3():
    from evaluation.detection_accuracy_eval import eval_accuracy
    from config import Config
    anno = Config.COCO_ANN_VAL_FILE
    # from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
    # model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1)
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    result = eval_accuracy(model, 224, 100, device)
    print(result)

def test_search_res50():
    from models.backbone.ofa_supernet import get_ofa_supernet_resnet50
    from arch_search.arch_search_ofa_resnet50 import create_study, run_study
    study = create_study("test_search_resnet50")
    model = get_ofa_supernet_resnet50()
    run_study(model, study, 100, 'cuda', [240, 360, 480])

def test_search_res50_fcos():
    from arch_search.arch_search_ofa_resnet50_fcos import create_study, run_study
    study = create_study("test_search_resnet50_fcos")
    model = torch.load('ofa_resnet50_fcos.pth')
    run_study(model, study, 100, 'cuda', [240, 360, 480])

def test_search_mbv3_w12():
    from arch_search.arch_search_ofa_mbv3_w12 import create_study, run_study
    from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
    study = create_study("test_search_mbv3")
    model = get_ofa_supernet_mbv3_w12()
    run_study(model, study, 100, 'cuda', [240, 360, 480])

def test_search_mbv3_w12_fcos():
    from arch_search.arch_search_ofa_mbv3_w12_fcos import create_study, run_study
    # 限制显存到2GB (Nano的显存大小)
    torch.cuda.set_per_process_memory_fraction(0.25)  # 3070 8GB的25%约等于2GB   
    study = create_study("test_search_mbv3_fcos")
    model = torch.load('ofa_mbv3_w12_fcos.pth')
    run_study(model, study, 200, 'cuda', [640])

def test_pareto_front():
    import optuna
    exp = optuna.load_study(study_name="test_search_mbv3_fcos", storage="sqlite:///test_search_mbv3_fcos.db")
    for t in exp.best_trials:
        print(t.number, t.values)

def test_train_subnet():
    from train.train_detection_subnet import train_subnet
    from models.detection.ofa_mbv3_w12_fcos import get_ofa_mbv3_w12_fcos_model, load_pretrained_fcos, set_training_params
    model = get_ofa_mbv3_w12_fcos_model()
    import torch
    import os
    if os.path.exists('ofa_mbv3_w12_fcos_subnet.pth'):
        model = torch.load('ofa_mbv3_w12_fcos_subnet.pth')
        print('Load model from ofa_mbv3_w12_fcos_subnet.pth')
    load_pretrained_fcos(model)
    set_training_params(model)
    from models.backbone.ofa_supernet import get_min_net_config
    min_net_config = get_min_net_config(ofa_supernet_name='ofa_supernet_mbv3_w12')
    train_subnet(model, subnet_config=min_net_config, num_epochs=20, save_path='ofa_mbv3_w12_fcos_subnet.pth', batch_size=4,
                backbone_learning_rate=1e-4, 
                head_learning_rate=1e-3,
                min_backbone_lr=1e-5, 
                min_head_lr=1e-4)
    
def test_train_subnet2():
    from train.train_detection_subnet import train_subnet
    from models.detection.ofa_resnet50_fcos import get_ofa_resnet50_fcos_model, load_pretrained_fcos, set_training_params
    model = get_ofa_resnet50_fcos_model()
    import torch
    import os
    if os.path.exists('ofa_resnet50_fcos_subnet.pth'):
        model = torch.load('ofa_resnet50_fcos_subnet.pth')
        print('Load model from ofa_resnet50_fcos_subnet.pth')
    load_pretrained_fcos(model)
    set_training_params(model)
    from models.backbone.ofa_supernet import get_min_net_config
    min_net_config = get_min_net_config(ofa_supernet_name='ofa_supernet_resnet50')
    train_subnet(model, subnet_config=min_net_config, num_epochs=20, save_path='ofa_resnet50_fcos_subnet.pth', batch_size=4,
                backbone_learning_rate=1e-3, 
                head_learning_rate=1e-2,
                min_backbone_lr=1e-5, 
                min_head_lr=1e-4)
        
def eval_net_acc():
    from train.train_detection_subnet import train_subnet
    from models.detection.ofa_mbv3_w12_fcos import get_ofa_mbv3_w12_fcos_model, load_pretrained_fcos, set_training_params
    model = get_ofa_mbv3_w12_fcos_model()
    import torch
    import os
    if os.path.exists('ofa_mbv3_w12_fcos_subnet.pth'):
        model = torch.load('ofa_mbv3_w12_fcos_subnet.pth')
        print('Load model from ofa_mbv3_w12_fcos_subnet.pth')    

if __name__ == '__main__':
    # train_fcos_mbv3_w12()
    # train_fcos_resnet50()
    train_fasterrcnn_mbv3_w12()
    # train_fasterrcnn_resnet50()
    # test_calib_bs()
    # test_classification_api()
    # test_det_api()
    # test_model_api()
    # test_fpn_with_other_det()
    # some_test3()
    # test_search_res50()
    # test_search_res50_fcos()
    # test_search_mbv3_w12()
    # test_search_mbv3_w12_fcos()
    # test_pareto_front()
    # test_train_subnet()
    # test_train_subnet2()

import torch

def test1():
    from models.backbone.ofa_supernet import get_ofa_supernet_resnet50
    model = get_ofa_supernet_resnet50()
    subnet = model.sample_active_subnet()
    print(subnet)

def test2():
    from models.backbone.ofa_supernet import get_architecture_dict
    print(get_architecture_dict("ofa_supernet_resnet50"))
    print(get_architecture_dict("ofa_supernet_mbv3_w10"))
    print(get_architecture_dict("ofa_supernet_mbv3_w12"))

def test_fpn1():
    from models.fpn.ofa_supernet_mbv3_w12_fcos_fpn import Mbv3W12FcosFpn
    from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
    mbv3_w12_fcos_fpn = Mbv3W12FcosFpn(get_ofa_supernet_mbv3_w12())
    x = torch.rand(1, 3, 224, 224)
    out = mbv3_w12_fcos_fpn(x)
    print(out.keys())
    print(out["0"].shape)
    print(out["1"].shape)
    print(out["2"].shape)

def test_fcos1():
    from models.detection.ofa_mbv3_w12_fcos import get_ofa_mbv3_w12_fcos_model
    model = get_ofa_mbv3_w12_fcos_model()
    model.train()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x = torch.rand(1, 3, 224, 224).to(device)
    targets = [{
        'boxes': torch.tensor([[100, 100, 200, 200]], device=device),
        'labels': torch.tensor([1], device=device),
    }]
    model.to(device)
    for i in range(10):
        print
        ofa_network = model.backbone.body
        subnet_config = ofa_network.sample_active_subnet()
        ofa_network.set_active_subnet(**subnet_config)
        # test forward
        outputs = model(x, targets)
        print(f"Forward pass successful for subnet {i+1}")
        # test backward
        total_loss = sum(loss for loss in outputs.values())
        total_loss.backward()
        print(f"Backward pass successful for subnet {i+1}")
        torch.save(model, 'test_model.pth')

def test_full_model1():
    from models.fpn.ofa_supernet_mbv3_w12_fcos_fpn import Mbv3W12FcosFpn
    from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
    from torchvision.models.detection import FCOS
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = FCOS(Mbv3W12FcosFpn(get_ofa_supernet_mbv3_w12()), 91)
    model_path = 'test_model.pth'
    model = torch.load(model_path)
    model.eval().to(device)
    x = torch.rand(1, 3, 224, 224).to(device)
    for i in range(10):
        model(x)
        print(f"Inference successful for iteration {i+1}")

def test_bn_calibration1():
    from models.fpn.ofa_supernet_mbv3_w12_fcos_fpn import Mbv3W12FcosFpn
    from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
    from torchvision.models.detection import FCOS
    import torchvision.transforms as transforms
    
    model = FCOS(Mbv3W12FcosFpn(get_ofa_supernet_mbv3_w12()), 91)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    from utils.bn_calibration import set_running_statistics
    from datasets.calib_dataset import get_calib_dataloader
    for i in range(10):
        set_running_statistics(model, get_calib_dataloader(), 10)
        print(f"Calibration successful for iteration {i+1}")

def test_forward():
    from models.fpn.ofa_supernet_mbv3_w12_fcos_fpn import Mbv3W12FcosFpn
    from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
    from torchvision.models.detection import FCOS
    model = FCOS(Mbv3W12FcosFpn(get_ofa_supernet_mbv3_w12()), 91)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    from datasets.coco_dataset import get_test_dataset, get_dataloader
    calibration_dataloader = get_dataloader(get_test_dataset(), 1)

    from utils.bn_calibration import set_running_statistics
    for i in range(10):
        times = 0
        for data in calibration_dataloader:
            if data is None:
                continue
            times += 1
            if times > 10:
                break
            images, targets = data
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images, targets)
        print(f"forward successful for iteration {i+1}")

def test_det():
    from train.train_detection_networks import train
    from models.detection.ofa_mbv3_w12_fcos import get_ofa_mbv3_w12_fcos_model, load_pretrained_fcos, set_training_params
    model = get_ofa_mbv3_w12_fcos_model()
    load_pretrained_fcos(model)
    set_training_params(model)
    train(model, 10, 'test.pth')
    model = torch.load('test.pth')


if __name__ == '__main__':
    # test_fcos1()
    # test_full_model1()
    # test_bn_calibration1()
    # test_forward()
    test_det()
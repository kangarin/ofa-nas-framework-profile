import torch

def train_fcos_mbv3_w12():
    from train.train_detection_networks import train
    from models.detection.ofa_mbv3_w12_fcos import get_ofa_mbv3_w12_fcos_model, load_pretrained_fcos, set_training_params
    model = get_ofa_mbv3_w12_fcos_model()
    load_pretrained_fcos(model)
    set_training_params(model)
    train(model, 5, 'ofa_mbv3_w12_fcos.pth', batch_size=2)
    model = torch.load('ofa_mbv3_w12_fcos.pth')

if __name__ == '__main__':
    train_fcos_mbv3_w12()
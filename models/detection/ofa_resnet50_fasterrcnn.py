from models.fpn.ofa_supernet_resnet50_fasterrcnn_fpn import Resnet50FasterRcnnFpn
from models.backbone.ofa_supernet import get_ofa_supernet_resnet50
from torchvision.models.detection import FasterRCNN
from models.backbone.backbone_cleanup import cleanup_backbone
from utils.logger import setup_logger
logger = setup_logger('model')

def get_ofa_resnet50_fasterrcnn_model(num_classes = 91):
    '''
    这里backbone权重是预训练的supernet，其他层权重未训练
    '''
    backbone = get_ofa_supernet_resnet50()
    backbone = cleanup_backbone(backbone)
    model = FasterRCNN(Resnet50FasterRcnnFpn(backbone), num_classes)
    return model

def load_pretrained_fasterrcnn(model):
    '''
    可选的从fasterrcnn_resnet50_fpn预训练权重中加载head匹配层的权重，加速训练
    '''
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    pretrained_fasterrcnn_fpn = fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    for key in pretrained_fasterrcnn_fpn.state_dict().keys():
        if key in model.state_dict().keys():
            # 判断形状是否一致
            if model.state_dict()[key].shape != pretrained_fasterrcnn_fpn.state_dict()[key].shape:
                # logger.info('Shape not match ', key)
                continue
            model.state_dict()[key].copy_(pretrained_fasterrcnn_fpn.state_dict()[key])
        else:
            # logger.info('key not in ofa_fasterrcnn: ', key)
            pass
    logger.info('Pretrained weights loaded.')

def set_training_params(model, is_backbone_body_need_training = False,
                        is_backbone_dynamic_convs_need_training = True,
                        is_backbone_fpn_need_training = False,
                        is_head_need_training = True):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.backbone.body.parameters():
        param.requires_grad = is_backbone_body_need_training

    for param in model.backbone.dynamic_convs.parameters():
        param.requires_grad = is_backbone_dynamic_convs_need_training

    for param in model.backbone.fpn.parameters():
        param.requires_grad = is_backbone_fpn_need_training

    for param in model.rpn.parameters():
        param.requires_grad = is_head_need_training

    for param in model.roi_heads.parameters():
        param.requires_grad = is_head_need_training

    # 打印出需要训练的参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
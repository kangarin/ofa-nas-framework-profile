from models.fpn.ofa_supernet_resnet50_fcos_fpn import Resnet50FcosFpn
from models.backbone.ofa_supernet import get_ofa_supernet_resnet50
from torchvision.models.detection import FCOS

from utils.logger import setup_logger
logger = setup_logger('model')

def get_ofa_resnet50_fcos_model(num_classes = 91):
    '''
    这里backbone权重是预训练的supernet，其他层权重未训练
    '''
    model = FCOS(Resnet50FcosFpn(get_ofa_supernet_resnet50()), num_classes)
    return model

def load_pretrained_fcos(model):
    '''
    可选的从fcos_resnet50_fpn预训练权重中加载head匹配层的权重，加速训练
    '''
    from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
    pretrained_fcos_fpn = fcos_resnet50_fpn(weights = FCOS_ResNet50_FPN_Weights.COCO_V1)
    for key in pretrained_fcos_fpn.state_dict().keys():
        if key in model.state_dict().keys():
            model.state_dict()[key].copy_(pretrained_fcos_fpn.state_dict()[key])
        else:
            # logger.info('key not in ofa_fcos: ', key)
            pass
    logger.info('Pretrained weights loaded.')

# ofa_resnet50的backbone中fpn可以不用微调，因为维数和预训练权重能对上
# 根本原因在于mbv3中间stage输出的最大维数较低，是从[48, 136, 192]映射到256
# 而resnet50中间stage输出的最大维数与预训练模型一致，是从[512, 1024, 2048]映射到256
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

    for param in model.head.parameters():
        param.requires_grad = is_head_need_training

    # 打印出需要训练的参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
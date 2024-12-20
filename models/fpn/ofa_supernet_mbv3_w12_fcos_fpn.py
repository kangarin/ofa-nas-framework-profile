from collections import OrderedDict
from typing import Dict
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7, LastLevelMaxPool
from torch import nn, Tensor

from models.ops.dynamic_conv2d import DynamicConv2d

class Mbv3W12FcosFpn(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        # in_channels_list = [48, 136, 192]
        in_channels_list = [48, 96, 136, 192]
        out_channels = 192
        # extra_blocks = LastLevelP6P7(192, 192)
        extra_blocks = LastLevelMaxPool()
        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks
        )
        self.out_channels = out_channels

        # 添加bn层
        self.dynamic_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(len(in_channels_list)):
            self.dynamic_convs.append(
                DynamicConv2d(in_channels_list[i], in_channels_list[i], kernel_size=1)
            )
            self.bns.append(nn.BatchNorm2d(in_channels_list[i]))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        mid_features = OrderedDict()
        x = self.body.first_conv(x)
        x = self.body.blocks[0](x)
        stage_idx = 0
        for stage_id, block_idx in enumerate(self.body.block_group_info):
            depth = self.body.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.body.blocks[idx](x)
                if idx == active_idx[-1]:
                    # if stage_id == 0 or stage_id == 2:
                    if stage_id == 0:
                        pass
                    else:
                        feat = self.dynamic_convs[stage_idx](x)
                        feat = self.bns[stage_idx](feat)
                        mid_features[str(stage_idx)] = feat
                        stage_idx += 1
        x = self.fpn(mid_features)
        return x
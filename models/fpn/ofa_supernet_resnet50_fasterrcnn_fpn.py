from collections import OrderedDict
from typing import Dict
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torch import nn, Tensor

from models.ops.dynamic_conv2d import DynamicConv2d

class Resnet50FasterRcnnFpn(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        in_channels_stage2 = 256
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in range(1, 5)]
        out_channels = 256
        extra_blocks = LastLevelMaxPool()
        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks
        )
        self.out_channels = out_channels
        self.dynamic_convs = nn.ModuleList()
        for i in range(4):
            self.dynamic_convs.append(DynamicConv2d(in_channels_list[i], in_channels_list[i], kernel_size=1))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        for layer in self.body.input_stem:
            x = layer(x)
        x = self.body.max_pooling(x)
        mid_features = OrderedDict()
        for stage_id, block_idx in enumerate(self.body.grouped_block_index):
            depth_param = self.body.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                x = self.body.blocks[idx](x)
                if idx == active_idx[-1]:
                    mid_features[str(stage_id)] = self.dynamic_convs[stage_id](x)                
        x = self.fpn(mid_features)
        return x
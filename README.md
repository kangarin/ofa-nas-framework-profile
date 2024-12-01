# 说明
这个项目主要包括：
- ofa相关下游任务模型的实现和训练
- 在不同设备上进行架构搜索

基础模型参考：https://pytorch.org/hub/pytorch_vision_once_for_all/

```
import torch
super_net_name = "ofa_supernet_mbv3_w10" 
# other options: 
#    ofa_supernet_resnet50 / 
#    ofa_supernet_mbv3_w12 / 
#    ofa_supernet_proxyless

super_net = torch.hub.load('mit-han-lab/once-for-all', super_net_name, pretrained=True).eval()
```

需要修改config.py中数据集的路径：
- imagenet-mini：https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000  注意，可能需要额外下载ISVRC2012_devkit_t12.tar.gz
- COCO2017：https://cocodataset.org/#download 下载train val 和 annotations
- calib：可以用验证集，用作子网络bn层的calibration
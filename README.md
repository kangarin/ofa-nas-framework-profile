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
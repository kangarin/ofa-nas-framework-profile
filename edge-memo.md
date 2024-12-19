在边缘节点部署的一些坑：
1. python版本是3.6.9，注意安装对应版本的torch，torchvision，要两个一起安装，不然会报错
2. optuna只能安装旧版本，而且不能直接安装，否则会numpy、scipy报错，事实上系统中已经有了这两个库，但是还是会报错，必须用--no-deps参数安装，再把optuna的其他依赖项一个个安装
3. gfw的缘故无法访问gdrive等，必须先把模型和项目下载到本地，然后用如下代码加载模型：torch.hub.load('/home/nvidia/dwy/ofa-nas/.torch_hub_cache/mit-han-lab_once-for-all_master', 'ofa_supernet_mbv3_w10', pretrained=True, force_reload=False, source='local')

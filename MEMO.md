## TODO
1. det模型中可能包含一些无用结构，比如classifier，虽然未参与forward但可能占用内存，考虑删除
2. 应当为不同的设备写一个配置文件，里面包括推理设备（cpu/gpu，有gpu选gpu）、batch size可选列表（过大会卡死，应当有个上限）、image size可选列表（过大会卡死，应当有个上限）这些，然后再去做架构搜索

## MEMO
1. calib数据集不需要gt，只需要一个图片文件夹即可，测量函数，入参模型不需要train模式，eval即可，bn会在代码里做替换，已实测
2. 检测不需要resize图像，因为自定义的collate fn里面batch是返回一个列表，但是分类需要统一batch里的图像尺寸
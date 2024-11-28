## TODO
1. 目前transform尚未规范化，需要确保训练、测量、推理的transform一致，比如归一化


## MEMO
1. calib数据集不需要gt，只需要一个图片文件夹即可，测量函数，入参模型不需要train模式，eval即可，bn会在代码里做替换，已实测
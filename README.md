# AvmPept
环视相机全景图，停车位检测、可行使区域检测、多任务感知；后续再增加用于定位和建图的元素，如减速带、车道线、道路箭头。

(备注：开源模型，开源数据，开源框架，**非量产、非工程方案**)


### 1 依赖
data_loader, model用**pytorch**搭建，训练测试runner用**mmengine**搭建。

- 创建conda环境，安装pytorch（其它版本应该也行）：  
**conda create -n avmpept python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y**
- mmengine安装：**pip install -e . -v**
- labelme安装：**pip install labelme**



### 2 代码简介
1） tools文件夹：单任务和多任务的训练、测试和推理脚本，任务包含停车位检测(pld)、可行使区域检测(freespace)和多任务(multi_task)。

2） projects文件夹：包含数据处理(dataset)、模型(model)、损失函数(loss)和指标(metric)

### 3 方案简介(持续优化)
（非常感谢**博登智能**分享提供的环视泊车数据集，本人仅在学习中使用）
- backbone+neck：resnet18, FPN,  尝试使用ASPP(语义分割deeplabv3)
- pld_head：车位类型，角点检测，进入线，分割线检测，focal_loss, L1_loss
- fs_head：语义分割，bce_loss, lovasz_loss

### 4 测试
测试数据：https://zhuanlan.zhihu.com/p/564718292

权重下载：待上传

视频效果：
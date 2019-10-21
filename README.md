# mmdetection-new
一、mmdetection的安装（目前官方只支持Linux系统安装）

1，安装环境：

  Ubuntu16.04
  CUDA9.0
  CUDNN8.0
  Pytorch1.1
  Python3.7
2，这里建议使用单独的虚拟环境：

1，终端创建虚拟环境：conda create -n  虚拟环境名称  python=3.7

2，进入虚拟环境： source activate 虚拟环境名称 （退出指令：source deactivate）

3，利用conda 安装pytorch 和torchvision

conda install pytorch torchvision -c pytorch

4，下载mmdetection工具并配置虚拟环境:

 下载地址：https://github.com/python-bookworm/mmdetection-new

5，安装依赖库：

pip install -r requirements.txt

二、配置cascade_rcnn训练数据：

1，数据准备：

训练的原始样本图片放到mmdetection/data/coco/train2017文件夹下；用于验证的原始图片数据放data/coco/val2017

训练的标注数据（xml）放到mmdetection/data/coco/annotations/train文件夹；用于验证的标准数据（xml）放

mmdetection/data/coco/annotations/val文件夹；

2，xml转json:

打开mmdetection/xml2json.py，修改convert()中的文件路径，运行xml2json.py；

3，网络参数配置：cascade_rcnn_r50_fpn_1x.py

打开mmdetection/configs/cascade_rcnn_r50_fpn_1x.py：

     1，修改num_classes:#分类器的类别数量+1，+1是多了一个背景的类型；

     2，修改data_root:#数据集根目录

     3，修改img_scale:#输入的图像尺寸

     4，修改imgs_per_gpu:#每个gpu计算的图像数量

     5，修改workers_per_gpu:#每个gpu分配的线程数

     6，修改ann_file和img_prefix:#数据集annotation路径和数据集路径

     7，修改optimizer中的lr:#学习率，计算公式：imgs_per_gpu*0.00125

     8，修改total_epochs:#训练轮数

     9，work_dir:#log文件和模型文件存储路径

4，打开mmdetection/mmdet/datasets/coco.py:

     修改CLASSES中类别：#修改成自己的类别

三、运行train.py开始训练模型：

终端运行指令：python tools/train.py  configs/cascade_rcnn_r50_fpn_1x.py
# 神经网络和深度学习课程-期末作业任务2：训练图像分类模型

## 仓库介绍

本项目为课程DATA620004——神经网络和深度学习期末作业任务2的代码仓库

* 任务2：在CIFAR-100数据集上比较基于Transformer和CNN的图像分类模型

* 基本要求：

  （1） 分别基于CNN和Transformer架构实现具有相近参数量的图像分类网络；

  （2） 在CIFAR-100数据集上采用相同的训练策略对二者进行训练，其中数据增强策略中应包含CutMix；
  
  （3） 尝试不同的超参数组合，尽可能提升各架构在CIFAR-100上的性能以进行合理的比较。


### 文件说明
```bash
├── data/                                   # data的路径
├── model/                                  # 模型架构的代码
    ├── __init__.py
    ├── cnn.py                              # CNN模型
    └── transformer.py                      # ViT模型
├── runs/                                   # 模型训练过程记录
    ├── cnn_2024-06-26_22-25-19/            # 最优CNN模型的训练过程
    └── transformer_2024-06-26_22-25-32     # 最优ViT模型的训练过程
├── save_model_path/                        # 模型储存地址
    ├── cnn/                                # CNN模型权重
    └── transformer/                        # ViT模型权重
├── yaml/                                   # 模型训练的config路径
    ├── cnn.yaml                            # CNN模型的参数
    └── transformer.yaml                    # ViT模型的参数
├── __init__.py
├── cutmix.py                               # 数据增强策略CutMix的实现代码
├── data_download.py                        # data_loader函数
├── para.md                                 # 模型的参数记录
├── test.py                                 # 模型测试
├── train.py                                # 模型训练
├── README.md                               # 本文件
```

## Requirements

```bash
# Step 1. Create a conda environment and activate it.
conda create --name cvfinal python=3.10 -y
conda activate cvfinal

# Step 2. Install PyTorch following official instructions, e.g.
conda install pytorch torchvision -c pytorch

# Step 3. Install required packages
pip3 install argparse
pip3 install tensorboard
pip3 install matplotlib
```

## 一、下载数据集
执行以下指令下载`Cifar100`数据集，会自动保存在`data/`路径下

```bash
python data_download.py
```

## 二、模型训练和测试

### 1. 模型训练（Train）
模型的架构设置编写在`yaml`文件夹的对应文件中

运行以下指令进行训练
```bash
# CNN
python train.py --config yaml/cnn.yaml
# ViT
python train.py --config yaml/transformer.yaml
```

使用tensorboard查看对应的训练过程
```bash
tensorboard --logdir runs/
```

### 2. CutMix数据增强
* Cutmix 数据增强的代码在`cutmix.py`中
* 训练时，我们对一定数量的训练集进行Cutmix增强，具体实现代码如下，适配在`train.py`中：
```python
r = np.random.rand(1)
if r[0] < cutmix_prob:
    inputs, (targets_a, targets_b, lam) = cutmix(inputs, labels, alpha)
    outputs = model(inputs)
    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
else:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

### 3. 模型测试（Test）
模型权重地址：[https://pan.baidu.com/s/1ieCovzZTSRUV-rwjFFqGqg?pwd=5hcf](https://pan.baidu.com/s/1ieCovzZTSRUV-rwjFFqGqg?pwd=5hcf)

将下载的模型权重移至`save_model_path/`下，再运行以下命令可以返回在测试集上的表现
```bash
# CNN
python test.py --config yaml/cnn.yaml
# ViT
python test.py --config yaml/transformer.yaml
```

## 更多的细节详见报告
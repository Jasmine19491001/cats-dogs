# 猫狗分类：ResNet18 迁移学习基线

Day36 学习项目：使用 PyTorch 和 torchvision 的预训练 ResNet18，训练一个猫狗二分类头。

## 当前功能

- 使用 ImageFolder 从文件夹生成标签：cats=0，dogs=1。
- 使用预训练权重对应的图片预处理。
- 冻结原模型参数，替换并训练新的全连接分类头。
- 训练 1 个 Epoch，输出平均损失和准确率。
- 从原有 1000 张 validation 图片中固定划分 800 张验证、200 张测试。
- 在验证集上评估并保存 cats_dogs_baseline.pth。

测试集保留到后续最终评估，当前脚本不计算测试集指标。

## 下载完整项目

本仓库的 Releases 附件 cats_dogs_day36_full.zip 包含整理后的代码、3000 张数据集图片和用户已有的 cats_dogs_baseline.pth。解压后可直接在配置好的 PyTorch 环境中运行。

已有模型来自用户原始本地训练；代码修正版尚未重新训练。

## 环境与运行

在已配置好的 PyTorch 环境中运行；如需安装基础依赖：

```bash
python -m pip install -r requirements.txt
```

CUDA 用户需安装与机器匹配的 PyTorch/torchvision 版本。本依赖清单未锁定用户电脑上的实际版本。

将 cats_and_dogs_filtered 小数据集放在项目的 data 文件夹中，目录结构如下：

```text
cats_dogs/
├── cats_dogs.py
├── requirements.txt
├── README.md
└── data/
    └── cats_and_dogs_filtered/
        ├── train/
        │   ├── cats/  # 1000 张
        │   └── dogs/  # 1000 张
        └── validation/
            ├── cats/  # 500 张
            └── dogs/  # 500 张
```

数据集下载入口：https://download.mlcc.google.com/mledu-datasets/cats_and_dogs_filtered.zip

```bash
python cats_dogs.py
```

首次运行需要下载预训练权重。程序优先使用 CUDA，否则使用 CPU。数据路径以脚本所在目录为基准，不依赖电脑盘符。

## 训练设置

| 项目 | 设置 |
| --- | --- |
| 模型 | ResNet18，ResNet18_Weights.DEFAULT |
| 可训练部分 | 新分类头 Linear(512, 2) |
| 损失函数 | CrossEntropyLoss |
| 优化器 | Adam，学习率 0.001 |
| Batch size | 16 |
| Epoch | 1 |
| 训练图片 | 2000 |
| 验证 / 测试 | 800 / 200，划分种子 42 |

保持骨干为 eval 模式以固定 BatchNorm 统计量，分类头设为 train 模式。验证时同时使用 model.eval() 和 torch.no_grad()。

只固定了验证/测试划分种子，分类头初始化和训练顺序仍可能变化，因此每次运行的指标不保证相同。

## 已观察结果与保存格式

用户 Day36 本地运行截图显示：验证集 800 张，平均 Loss 约 0.05248，准确率 98.50%。此结果来自原始本地运行；上传前修正了冻结参数语句中的 requires_gard 拼写错误，修正版尚未重新训练，不将历史指标当作重新运行结果。

保存内容：model_state_dict、class_to_idx 和 split_seed。该文件是这一轮训练结果，不是多轮比较选出的最佳模型。

原始代码的优化器仅包含分类头参数，所以拼写错误虽导致骨干记录梯度，optimizer.step() 仍只更新分类头参数。

图片数据、压缩包和生成的模型权重保存在 Releases 的完整项目附件中，普通源码提交仅保留代码和运行说明。也可按以上说明准备数据并重新训练生成模型。

## 后续学习

- 多轮训练与损失、准确率曲线。
- 按验证集 Loss 保存最佳模型。
- 模型重载、单张图片预测和最终测试集评估。



## 项目简介

本项目基于 **U-Net** 实现血管图像的分割与评估，主要流程包括：\
1. 数据预处理\
2. 模型预测\
3. 结果评估（Dice、Jaccard、F1 Score）

------------------------------------------------------------------------

## 环境依赖

``` bash
pip install torch torchvision opencv-python tqdm numpy
```

------------------------------------------------------------------------

## 使用流程

### 1. 数据整理

``` bash
python Bourbon_extract_png.py
```

整理原始数据到 `predict_merge` 和 `predict_green` 文件夹。

### 2. 模型预测

``` bash
python Bourbon_predict.py
```

使用训练好的模型生成红/绿通道分割结果，输出到 `output_merge6/`。

### 3. 精度计算

``` bash
python Bourbon_calculate_accuracy.py
```

运行推理并保存预测结果。

### 4. 结果评估

``` bash
python Bourbon_cal_dice_jac.py
```

计算 **Dice、Jaccard、F1 Score** 并输出平均值。

------------------------------------------------------------------------

## 项目结构

    ├── Bourbon_extract_png.py        # 数据整理
    ├── Bourbon_predict.py            # 模型预测
    ├── Bourbon_calculate_accuracy.py # 精度计算
    ├── Bourbon_cal_dice_jac.py       # 指标评估
    ├── Bourbon_data_pre.py           # 数据集定义
    └── Bourbon_tiqushuju/            # 数据与模型目录

# Scene-Text-Detection-and-Recognition-benchmark

## Introduction
Scene Text Detection and Recognition is an open source scene text detection and recognition benchmark based on PyTorch

## Major features

- Various backbones and pretrained models
- Large-scale training configs
- High efficiency and extensibility

## Requirements

This is my experiment eviroument
- python3.6
- pytorch1.6.0+cu101

## Todo list
- [ ] implement DB
- [ ] implement CRNN
- [ ] implement ABCNet

## Dataset

### Detection
Supported:
- [x] ICDAR15
- [x] ICDAR17MLT
- [x] sythtext800K
- [x] Total Text
- [x] CTW1500
- [x] 2019ArT

### Recognition

## model zoo

Supported text detection:
- [x] EAST [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)
- [x] Psenet [Shape Robust Text Detection with Progressive Scale Expansion Network](https://arxiv.org/abs/1903.12473)
- [ ] DB [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf)

Supported text recognition:
- [ ] CRNN [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)

Supported End to End:


## Installation
Please refer to [install.md](docs/install.md) for installation and dataset preparation.

## Experiments
All models are trained in the same condition, and might not get the best result

### The performance of EAST
|Method|Backbone|Pretrain|Resolution|Dataset|Precision|Recall|F-score|FPS|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|EAST|VGG16|-|512|ICDAR15|0.81|0.81|0.81|-|
|EAST|VGG16|-|512|ICDAR17|0.72|0.61|0.66|-|
|EAST|VGG16|SynthText|512|ICDAR15|0.82|0.824|0.822|-|

### The performance of PSENet
|Method|Backbone|Pretrain|Resolution|Dataset|Precision|Recall|F-score|FPS|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|PSENet(1s)|ResNet50|-|640\*640|SynthText|-|-|-|-|
|PSENet(1s)|ResNet50|-|640\*640|ICDAR15|0.816|0.795|0.805|-|
|PSENet(1s)|ResNet50|-|640\*640|ICDAR17(val)|0.755|0.614|0.677|-|
|PSENet(1s)|ResNet50|-|640\*640|ICDAR17(test)|0.762|0.643|0.698|-|
|PSENet(1s)|ResNet50|-|640\*640|Total-Text|0.8255|0.7597|0.7913|3.0|
|PSENet(1s)|ResNet50|SynthText|640\*640|ICDAR15|0.864|0.835|0.850|-|
|PSENet(1s)|ResNet50|SynthText+ICDAR17|640\*640|ICDAR15|0.883|0.853|0.868|3.0|
|PSENet(1s)|ResNet50|SynthText|640\*640|Total-Text|0.834|0.781|0.807|3.0|


## links

EAST:  https://github.com/SakuraRiven/EAST

PSENet: https://github.com/WenmuZhou/PSENet.pytorch

DB: https://github.com/WenmuZhou/DBNet.pytorch

https://github.com/WenmuZhou/PytorchOCR

## Contact

Eamil: wwj123@zju.edu.cn
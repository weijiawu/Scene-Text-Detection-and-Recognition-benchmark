# Shape Robust Text Detection with Progressive Scale Expansion Network

## Requirements
* pytorch 1.1
* torchvision 0.3
* pyclipper
* opencv3
* gcc 4.9+

## Update 
### 20190401


### Download

## Data Preparation

## Train
ICDAR15   ResNet50 0.82

ICDAR17   ResNet50  
        eval  "precision": 0.7550, "recall": 0.6142959, "hmean": 0.677440      
        test "precision": 76.19%, "recall": 64.31%, "hmean": 69.75%    
        加上synthtext的pretrain：
        eval "precision": 0.7708, "recall": 0.61568 "hmean": 0.68458
        
TotalText  ResNet50   pre:0.8255  recall: 0.7597   f1: 0.7913
SynthText  
    训练预训练的时候注意 ICD15和ICD17使用的是6个kernel但是Total使用的是7个kernel(可以统一一下，问题也不大)


## Test

## Predict 





### reference
1. https://github.com/liuheng92/tensorflow_PSENet
2. https://github.com/whai362/PSENet

## SSIA

This is the mindspore official repository for SSIA


## Introduction

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10091166

## Requirement

* mindspore 2.2.11
* cuda 11.1
* cudnn 1.11 8.0.5.39
* PyYAML 6.0.1

## Data

Data set source:

https://github.com/ninghailong/Cross-Modal-Remote-Sensing-Image-Sound-Retrieval

```
data
├── rsicd_images
    ├── 00001.jpg
    ├── 00002.jpg
    ├── 00003.jpg
    ├── ...
    └── viaduct_9.jpg
├── rsicd_mat
    ├── test_audios.mat
    └── train_audios.mat
└── rsicd_precomp
    ├── test_auds.txt
    ├── test_caps.txt
    ├── test_filename.txt
    ├── train_auds.txt
    ├── train_auds_verify.txt
    ├── train_caps.txt
    ├── train_filename.txt
    ├── train_filename_verify.txt
    ├── val_auds_verify.txt
    └── val_filename_verify.txt

```
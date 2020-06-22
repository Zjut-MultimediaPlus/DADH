# Deep Adversarial Discrete Hashing for Cross-Modal Retrieval

## Introduction

This is the source code of ICMR 2020 paper "Deep Adversarial Discrete Hashing for Cross-Modal Retrieval".

## Requirements

- Python 3.5+
- Pytorch 1.0.0+

##  Train

```shell
python main.py train
```

## Test

```shell
python main.py test
```

## Datasets

- NUS-WIDE 
- MIRFlickr25K

The two datasets are provided by the authors of DCMH [Jiang, et al, CVPR'2017], which can be downloaded from their homepage.

## Framework

![](./framework.png)

## Result

<img src="./result.png" style="zoom: 67%;" />

## Note

Our codes were modified from the implementation of "Adversary Guided Asymmetric Hashing for Cross-Modal Retrieval", written by Wen Gu. Please cite the  two papers (AGAH and DADH) when you use the codes.

## Citing DADH & AGAH

```
@inproceedings{Bai2020,
  author={Cong Bai, Chao Zeng, Qing Ma, Jinglin Zhang and Shengyong Chen.},
  booktitle={Proceedings of the 2020 on International Conference on Multimedia Retrieval},
  pages={},
  title={Deep Adversarial Discrete Hashing for Cross-Modal Retrieval},
  year={2020},
}
```
```
@inproceedings{Gu2019,
author = {Gu, Wen and Gu, Xiaoyan and Gu, Jingzi and Li, Bo and Xiong, Zhi and Wang, Weiping},
booktitle = {Proceedings of the ACM International Conference on Multimedia Retrieval},
pages = {159--167},
title = {{Adversary guided asymmetric hashing for cross-modal retrieval}},
year = {2019}
}
```

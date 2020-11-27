# Deep Adversarial Discrete Hashing for Cross-Modal Retrieval

## Introduction

This is the source code of ICMR 2020 paper "Deep Adversarial Discrete Hashing for Cross-Modal Retrieval".

## Requirements

- Python 3.5+
- Pytorch 1.0.0+

##  Train

```shell
##before train##
python -m visdom.server
##train our method##
python main.py train
```

## Test

```shell
python main.py test
```

## Datasets
- CNN-F: https://pan.baidu.com/s/17R7t1qKNskIDWzLhALYFPg 提取码：cou4 
- NUS-WIDE: 
           Google: https://drive.google.com/file/d/125G-B7sIQPVIcRk4W7qR-tkc6gcBP-mX/view?usp=sharing <br>
           Baidu: comming soon
- MIRFlickr25K: 
           Google: https://drive.google.com/file/d/1Eca2meBpmhnfezkUVqQJo0tJnEpwQwi2/view?usp=sharing <br>
           Baidu: https://pan.baidu.com/s/14JrUH2AdnvDV1ezs0Qxc5w 提取码：22fs 

## Framework

![framework](framework.png)

## Result

<img src="result.png" style="zoom: 67%;" />

## Note

Our codes were modified from the implementation of "Adversary Guided Asymmetric Hashing for Cross-Modal Retrieval", written by Wen Gu. Please cite the  two papers (AGAH and DADH) when you use the codes.

## Citing DADH & AGAH

```
@inproceedings{Bai2020,
  author={Cong Bai, Chao Zeng, Qing Ma, Jinglin Zhang and Shengyong Chen.},
  booktitle={Proceedings of the 2020 on International Conference on Multimedia Retrieval},
  pages={525-531},
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

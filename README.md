# UCDA
This the the repository includes the official codes for our ICCV2021 paper "[Unsupervised Curriculum Domain Adaptation for No-Reference Video Quality Assessment](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Unsupervised_Curriculum_Domain_Adaptation_for_No-Reference_Video_Quality_Assessment_ICCV_2021_paper.html)".

![image](https://github.com/cpf0079/UCDA/blob/main/framework.png)

## Training
We use the 10-fold cross-validation in our experiments. To reach to the comparable performance you may need to train a few times. 

Step 1. Conduct the domain adaptation between source and target domains by running:
```
$ python ./Source/first_uda.py
```
Step 2. Uncertainty-based ranking to split target domain into subdomains by running:
```
$ python ./Source/ranking.py
```
Step 3. Conduct the domain adaptation between subdomains by running:
```
$ python ./Source/second_uda.py
```

## Environment
* Python 3.6.5
* Pytorch 1.0.1
* Cuda 9.0 Cudnn 7.1 

## Citation
If you find this work useful for your research, please cite our paper:
```
@InProceedings{Chen_2021_ICCV,
    author    = {Chen, Pengfei and Li, Leida and Wu, Jinjian and Dong, Weisheng and Shi, Guangming},
    title     = {Unsupervised Curriculum Domain Adaptation for No-Reference Video Quality Assessment},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {5178-5187}
}
```

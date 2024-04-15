#  ACMTOIS2023: The ITV model for Text-to-Video Retrieval (Ad-hoc video search)

This is the official source code of our ITV paper: [(Un)likelihood Training for Interpretable Embedding](https://arxiv.org/abs/2207.00282).

![architure](https://github.com/nikkiwoo-gh/ITV/blob/main/images/ITV.png)

## Environment

We used Anaconda to setup a workspace with PyTorch 1.8. Run the following script to install the required packages.

```shell
conda create -n ITV python==3.8 -y
conda activate ITV
git clone https://github.com/nikkiwoo-gh/ITV.git
cd ITV
pip install -r requirements.txt
```

### Stanford coreNLP server for concept bank construction
```shell
./do_install_StanfordCoreNLIP.sh
```

## Downloads

### Data

See the [data](data) page.



### Checkpoints

[ITV trained on tgif-msrvtt10k-VATEX](https://portland-my.sharepoint.com/:f:/g/personal/jiaxinwu9-c_my_cityu_edu_hk/Eo0j5dwNCZxHsy7cDmKYoEABIvLSZY53ikqDPa4sInMruA?e=ycxClq)

## Usages


### 1. build bag of word vocabulary and concept bank
```shell
./do_get_vocab_and_concept.sh $collection
```

e.g.,
```shell
./do_get_vocab_and_concept.sh tgif-msrvtt10k-VATEX
```

### 2. prepare the data
See the [data](data) page.

### 3. train ITV
```shell
./do_train_ITV.sh
```

### 4. Inference on TRECVid datasets
```shell
./do_predition_iacc.3_ITV.sh
./do_predition_v3c1_ITV.sh
./do_predition_v3c2_ITV.sh
```

### 5. Evalution
Remember to set the score_file correctly to your own path.
```shell
cd tv-avs-eval/
do_eval_iacc.3.sh
do_eval_v3c1.sh
do_eval_v3c2.sh
```

## Citation

```latex
@inproceedings{ACMTOIS2023_ITV,
author = {Wu, Jiaxin and Ngo, Chong-Wah and Chan, Wing-Kwong and Hou, Zhijian},
title = {(Un)likelihood Training for Interpretable Embedding},
year = {2023},
volume = {42},
number = {3},
journal = {ACM Transactions on Information Systems},
pages = {1-26},
}
```



## Contact

If you have any questions, please feel free to contact me

- Jiaxin Wu (jiaxin.wu@my.cityu.edu.hk)

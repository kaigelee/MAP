# MAP
Boosting Adversarial Transferability of Black-Box Attack

By Kaige Li, Maoxian Wan, Qichuan Geng, Xiaochun Cao, Senior Member, IEEE, and Zhong Zhou. 

The paper is under review. The full code will be released after review.


## Requirements
+ Python >= 3.6
+ PyTorch >= 1.12.1
+ Torchvision >= 0.13.1
+ timm >= 0.6.12

```bash
pip install -r requirements.txt
```


## Usage
We randomly sample 1,000 images from ImageNet validate set, in which each image is from one category and can be correctly classified by the adopted models (For some categories, we cannot choose one image that is correctly classified by all the models. In this case, we select the image that receives accurate classifications from the majority of models.). Download the data from [![GoogleDrive](https://img.shields.io/badge/GoogleDrive-space-blue)
](https://drive.google.com/file/d/1d-_PKYi3MBDPtJV4rfMCCtmsE0oWX7ZB/view?usp=sharing) or [![Huggingface Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/datasets/Trustworthy-AI-Group/TransferAttack/blob/main/data.zip) into `/path/to/data`. Then you can execute the attack as follows:

```
python main.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet18 --attack mifgsm --model=resnet18 --mask=True
python main.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet18 --eval
```

## Overview

Pending!

## Experimental Results

<p align="center">
  <img src="figs/results.png" alt="results-of-our-method" width="800"/></br>
  <span align="center">ASR ($\%$) of various transfer-based attacks against normally trained models, and their enhanced version by our method, using Res-18 as the surrogate model. </span> 
</p>
Our MAP achieves noticeable performance improvements on various black-box attack methods.



## TODO
- [ ] Upload the updated code
- [ ] Reorganize the code repository

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [SIA](https://github.com/xiaosen-wang/SIT)
* [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack)

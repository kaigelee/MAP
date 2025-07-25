# MAP
MAP: Masked Adversarial Perturbation for Boosting Black-box Attack Transferability

<img width="2414" height="70" alt="image" src="https://github.com/user-attachments/assets/c4698d6f-840b-4cc2-a635-4f9086c4f09c" />

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
python main.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet18 --attack mifgsm --model=resnet18 
python main.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet18 --eval
```

## Overview

<p align="center">
  <img src="figs/map.jpg" alt="results-of-our-method" width="800"/></br>
  <span align="center">Black-box attack with the proposed Masked Adversarial Perturbation (MAP). During iterations, we generate diverse soft masks with different mask
ratios according to iterations t, which masks several patches in the adversarial perturbation to reduce its complexity and improve its diversity. </span> 
</p>


### 🤗 Masked Adversarial Perturbation

To use our MAP to boost black-box attacks, you only need a few lines of code as demonstrated below.

```python
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

@torch.no_grad()
def generate_mask(self, imgs, ratio=0.7,patch_size=8):
    B, _, H, W = imgs.shape
    mshape = B, 1, round(H / patch_size), round(
                W / patch_size)
    input_mask = torch.rand(mshape, device=imgs.device)
    input_mask = (input_mask > ratio).float()
    input_mask = F.interpolate(input_mask, size=(H, W), mode='bilinear', align_corners=False)
    return input_mask
```

## Method Comparison

|              |                                                                                                                               PAR <br> (Patch-wise Adversarial Removal)                                                                                                                               |                     CISA <br> (Customized Iteration and Sampling Attack)                                                                                                                                                                                                                                                        |                                                                                                            MAP <br> (Masked Adversarial Perturbation)                                                                    |
|:------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   Objective  | Enhance query efficiency and noise compression in decision-based black-box attacks by removing noise from regions with low sensitivity in a patch-wise manner.                                                                                                                                                                         |   Optimize query efficiency in black-box attacks by integrating transfer-based and decision-based attacks, while adopting adaptive iteration and customized sampling for better noise compression.                                                                                                                                                                                                        |      Boost adversarial transferability across architectures (e.g., CNNs, ViTs and Hybrid Models) by randomly masking adversarial perturbations.                                        |
|   Core Idea  | - Splits the adversarial example into multiple patches and evaluate their noise sensitivity.<br>- Prioritize the removal of noise from low-sensitivity regions to reduce redundant perturbations. <br>- Employ a coarse-to-fine search process to refine noise compression and enhance query efficiency. |  - Bridge transfer-based and decision-based attacks for query-efficient black-box adversarial attack.<br>- Use Gaussian Stepsize Adjustment to adaptively set stepsize for transfer-based attack.<br>- Customize the sampling process and stepsize as well as mask to achieve efficient noise compression in decision-based attacks. <br>- Relax the transition function of CISA to accelerate noise compression. |  - Improve the transferability of adversarial examples by explicitly diversifying adversarial perturbations through random masks.<br>- Utilize Soft Mask Generation (SMG) to ensure smooth masking and reduce statistical shifts.<br> - Employ Curriculum Mask Learning (CML) to gradually increase the masking ratio, further enhancing generalization.  |
| Key Features | Focuses on better noise compression at the same number of queries via separately compressing the noise on each patch.                                                                                               | Focuses on better query efficiency and noise compression by bridging two attack strategies and customized iteration and sampling.                                                                                                     |   Focuses on boosting black-box attack transferability across different models via randomly masking adversarial perturbation.   |

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

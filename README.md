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

### 🤗 Transformers

To use Qwen-Chat for the inference, all you need to do is to input a few lines of codes as demonstrated below. Remember to pass in the correct model names or paths, such as "Qwen/Qwen-7B-Chat" and "Qwen/Qwen-14B-Chat". However, **please make sure that you are using the latest code.**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Model names: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    device_map="auto",
    trust_remote_code=True
).eval()

# Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# 1st dialogue turn
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。

# 2nd dialogue turn
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
# 这是一个关于一个年轻人奋斗创业最终取得成功的故事。
# 故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。从小，李明就立下了一个目标：要成为一名成功的企业家。
# 为了实现这个目标，李明勤奋学习，考上了大学。在大学期间，他积极参加各种创业比赛，获得了不少奖项。他还利用课余时间去实习，积累了宝贵的经验。
# 毕业后，李明决定开始自己的创业之路。他开始寻找投资机会，但多次都被拒绝了。然而，他并没有放弃。他继续努力，不断改进自己的创业计划，并寻找新的投资机会。
# 最终，李明成功地获得了一笔投资，开始了自己的创业之路。他成立了一家科技公司，专注于开发新型软件。在他的领导下，公司迅速发展起来，成为了一家成功的科技企业。
# 李明的成功并不是偶然的。他勤奋、坚韧、勇于冒险，不断学习和改进自己。他的成功也证明了，只要努力奋斗，任何人都有可能取得成功。

# 3rd dialogue turn
response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
print(response)
# 《奋斗创业：一个年轻人的成功之路》
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

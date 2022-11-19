# DualPrompt PyTorch Implementation

This repository contains PyTorch implementation code for awesome continual learning method <a href="https://arxiv.org/pdf/2204.04799.pdf">DualPrompt</a>, <br>
Wang, Zifeng, et al. "DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning." ECCV. 2022.

The official Jax implementation is <a href="https://github.com/google-research/l2p">here</a>.

## Environment
The system I used and tested in
- Ubuntu 20.04.4 LTS
- Slurm 21.08.1
- NVIDIA GeForce RTX 3090
- Python 3.8

## Usage
First, clone the repository locally:
```
git clone https://github.com/Lee-JH-KR/dualprompt-pytorch
cd dualprompt-pytorch
```
Then, install the packages below:
```
pytorch==1.12.1
torchvision==0.13.1
timm==0.6.7
pillow==9.2.0
matplotlib==3.5.3
```
These packages can be installed easily by 
```
pip install -r requirements.txt
```

## Data preparation
If you already have CIFAR-100 or ImageNet-R, pass your dataset path to  `--data-path`.


The datasets aren't ready, change the download argument in `datasets.py` as follows

**CIFAR-100**
```
datasets.CIFAR100(download=True)
```

**ImageNet-R**
```
Imagenet_R(download=True)
```

## Training
To train a model via command line:

Single node with single gpu
```
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        <cifar100_dualprompt or imr_dualprompt> \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path /local_datasets/ \
        --output_dir ./output \
        --epochs 5
```

Single node with multi gpus
```
python -m torch.distributed.launch \
        --nproc_per_node=<Num GPUs> \
        --use_env main.py \
        <cifar100_dualprompt or imr_dualprompt> \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path /local_datasets/ \
        --output_dir ./output \
        --epochs 5
```

Also available in <a href="https://slurm.schedmd.com/documentation.html">Slurm</a> system by changing options on `train_cifar100_l2p.sh` or `train_five_datasets.sh` properly.

### Multinode train

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):

```
pip install submitit
```

To train a model on 2 nodes with 4 gpus each:

```
python run_with_submitit.py <cifar100_dualprompt or imr_dualprompt> --shared_folder <Absolute Path of shared folder for all nodes>
```

Absolute Path of shared folder must be accessible from all nodes.<br>
According to your environment, you can use `NCLL_SOCKET_IFNAME=<Your own IP interface to use for communication>` optionally.

## Evaluation
To evaluate a trained model:
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py <cifar100_dualprompt or imr_dualprompt> --eval
```

## Result
Test results on a single gpu.
### Split-CIFAR100
| Name | Acc@1 | Forgetting |
| --- | --- | --- |
| Pytorch-Implementation | 86.6 | 5.12 |
| Reproduce Official-Implementation | 85.59 | 5.03 |

### Split-ImageNet-R
| Name | Acc@1 | Forgetting |
| --- | --- | --- |
| Pytorch-Implementation | 68.06 | 4.89 |
| Reproduce Official-Implementation | 67.55 | 5.06 |

Here are the metrics used in the test, and their corresponding meanings:

| Metric | Description |
| ----------- | ----------- |
| Acc@1  | Average evaluation accuracy up until the last task |
| Forgetting | Average forgetting up until the last task |


## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Cite
```
@article{wang2022dualprompt,
  title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
  author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
  journal={European Conference on Computer Vision},
  year={2022}
}
```

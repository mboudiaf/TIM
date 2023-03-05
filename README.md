# TIM: Transductive Information Maximization


##  Introduction
This repo contains the code for our NeurIPS 2020 paper "Transductive Infomation Maximization (TIM) for few-shot learning" available at https://arxiv.org/abs/2008.11297. Our method maximizes the mutual information between the query features and predictions of a few-shot task, subject to supervision constraint from the support set. Results provided in the paper can be reproduced with this repo. Code was developped under python 3.8.3 and pytorch 1.4.0. The code is parallelized over tasks (which makes the execution of the 10'000 tasks very efficient).


## 1. Getting started


Please find the data and pretrained models at [icloud drive](https://www.icloud.com/iclouddrive/0f3PFO3rJK0fk0nkCe8sDKAiQ#TIM). Please use `cat` command to reform the original file, and extract 

**Checkpoints:** The checkpoints/ directory should be placed in the root dir, and be structured as follows:

```
├── mini
│   └── softmax
│       ├── densenet121
│       │   ├── best
│       │   ├── checkpoint.pth.tar
│       │   └── model_best.pth.tar
```

**Data:** The checkpoints should be placed in the root directory, and have a structure like. Because of the size, tiered_imagenet has been sharded into 24 shard, 1GB each. Use `cat tiered_imagenet_*` to reform the original file. Extract everything to data/. The data folder should should be structured as follows:

```
├── cub
  ├── attributes.txt
  └── CUB_200_2011
      ├── attributes
      ├── bounding_boxes.txt
      ├── classes.txt
      ├── image_class_labels.txt
      ├── images
      ├── images.txt
      ├── parts
      ├── README
      └── train_test_split.txt
├── mini_imagenet
└── tiered_imagenet
  ├── class_names.txt
  ├── data
  ├── synsets.txt
  ├── test_images_png.pkl
  ├── test_labels.pkl
  ├── train_images_png.pkl
  ├── train_labels.pkl
  ├── val_images_png.pkl
  └── val_labels.pkl
```

All required libraries should be easily found online, except for visdom_logger that you can download using:
```
pip install git+https://github.com/luizgh/visdom_logger
```

## 2. Train models (optional)

Instead of using the pre-trained models, you may want to train the models from scratch. Before anything, don't forget to activate the downloaded environment:
```python
source env/bin/activate
```
Then to visualize the results, turn on your local visdom server:
```python
python -m visdom.server -port 8097
```
and open it in your browser : http://localhost:8097/ . Then, for instance, if you want to train a Resnet-18 on mini-Imagenet, go to the root of the directory, and execute:
```python
bash scripts/train/resnet18.sh
```

**Important :** Whenever you have trained yourself a new model and want to test it, please specify the option `eval.fresh_start=True` to your test command. Otherwise, the code may use cached information (used to speed-up experiments) from previously used models that are longer valid.

## 3. Reproducing the main results

Before anything, don't forget to activate the downloaded environement:
```python
source env/bin/activate
```

### 3.1 Benchmarks (Table 1. in paper)


|(1 shot/5 shot)|     Arch    | mini-Imagenet | Tiered-Imagenet |
| 	   ---      |      ---    |      ---      |	   ---          |
| TIM-ADM       |   Resnet-18 | 73.6 / **85.0**  | **80.0** / **88.5** |
| TIM-GD        |   Resnet-18 |  **73.9** / **85.0**  | 79.9 / **88.5**  |
| TIM-ADM       |   WRN28-10  |  77.5 / 87.2  | 82.0 / 89.7     |
| TIM-GD        |   WRN28-10  |  **77.8** / **87.4**  | **82.1** / **89.8** |

To reproduce the results from Table 1. in the paper, use the bash files at scripts/evaluate/. For instance, if you want to reproduce the methods on mini-Imagenet, go to the root of the directory and execute:
```python
bash scripts/evaluate/<tim_adm or tim_gd>/mini.sh
```
This will reproduce the results for the three network architectures in the paper (Resnet-18/WideResNet28-10/DenseNet-121). Upon completion, exhaustive logs can be found in logs/ folder


### 3.2 Domain shift (Table 2. in paper)

|(5 shot)       |     Arch    |        CUB -> CUB     | mini-Imagenet -> CUB |
| 	   ---      |      ---    |        ---            |	       ---           |
| TIM-ADM       |   Resnet18  |         90.7          |        70.3          |
| TIM-GD        |   Resnet18  |       **90.8**        |      **71.0**        |

If you want to reproduce the methods on CUB -> CUB, go to the root of the directory and execute:
```python
bash scripts/evaluate/<tim_adm or tim_gd>/cub.sh
```
If you want to reproduce the methods on mini -> CUB, go to the root of the directory and execute:
```python
bash scripts/evaluate/<tim_adm or tim_gd>/mini2cub.sh
```

### 3.3 Tasks with more ways (Table 3. in paper)

If you want to reproduce the methods with more ways (10 and 20 ways) on mini-Imagenet, go to the root of the directory and execute:

```python
bash scripts/evaluate/<tim_adm or tim_gd>/mini_10_20_ways.sh
```

|(1 shot/5 shot)|    Arch     |       10 ways     |       20 ways        |
| 	   ---      |     ---     |        ---        |	       ---           |
| TIM-ADM       |   Resnet18  |   56.0 / **72.9** |  **39.5** / 58.8     |
| TIM-GD        |   Resnet18  |**56.1** / 72.8    |    39.3 / **59.5** |


### 3.4 Ablation study (Table 4. in paper)

If you want to reproduce the 4 loss configurations of on mini-Imagenet, Tiered-Imagenet and CUB, go to the root of the directory and execute:
```python
bash scripts/ablation/<tim_adm or tim_gd or tim_gd_all>/weighting_effect.sh
```
for respectively TIM-ADM, TIM-GD {W} and TIM-GD {phi, W}.



<img src="plots/mini.png" width="800" height="400"/>

## Contact

For further questions or details, reach out to Malik Boudiaf (malik.boudiaf.1@etsmtl.net)

## Acknowledgements

We would like to thank the authors from SimpleShot code https://github.com/mileyan/simple_shot and LaplacianShot https://github.com/imtiazziko/LaplacianShot for giving access to their pre-trained models and to their codes from which this repo was inspired.

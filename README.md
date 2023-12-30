# Universal Backdoor Attacks: Official PyTorch Implementation
![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg?style=plastic)
![PyTorch 1.13.1](https://img.shields.io/badge/torch-1.13.1-green.svg?style=plastic)

## Code Summary

All scripts to interact with this code from the command line are located in the ```/examples``` folder. 
We include the following scripts

* ```./examples/embed.py```: Trains a model on poisoned data.
* ```./examples/defend.py```: Evaluates a model repair defense on pre-trained models.
* ```./examples/detect.py```: Evaluates a backdoor detection method on pre-trained models.
* ```./examples/grid_parameter_search.py```: Outputs optimal hyper-parameters from a grid search.
* ```./examples/grid_robustness_evaluate.py```: Evaluates the robustness of an attack against a set of defenses. 
Plots the entire utility/integrity trade-off curve and caches all results in a database. 

> **Hint:** Make sure to check out the ```./configs``` folder to see the configurations we used in our paper.

## Get Started
All dependencies can be installed via pip via the following command.
```shell
$ pip install -r requirements.txt
```

### Embed a Backdoor from Scratch
Embed a badnet backdoor from scratch on CIFAR-10 using the following command.
```shell
$ cd examples
$ python embed.py --config ../configs/cifar10/attack/badnet/badnet.yml
```


### Defend against a Backdoor
We first evaluate the feature grinding defense against a pre-trained badnet model.
All model and configuration files will be downloaded automatically.
Run the following command.
```shell
$ cd examples
$ python defend.py --config ../configs/cifar10/defense/feature-grinding/badnet.yml 
```

## Implemented Methods
|                                              Attack                                              |     Type     | Modify Labels |     CIFAR-10      |     ImageNet      |
|:------------------------------------------------------------------------------------------------:|:------------:|:-------------:|:-----------------:|:-----------------:|
|                           [BadNets](https://arxiv.org/abs/1708.06733)                            |    Poison    |    &#9745;    |      &#9745;      |      &#9745;      |
|              [AdvCleanLabel](https://people.csail.mit.edu/madry/lab/cleanlabel.pdf)              |    Poison    |    &#9744;    |      &#9745;      |      &#9745;      |
|                            [Refool](https://arxiv.org/abs/2007.02343)                            |    Poison    |    &#9744;    |      &#9745;      |      &#9745;      |
|                   [Adaptive Blend](https://openreview.net/pdf?id=_wSHsgrVali)                    |    Poison    |    &#9745;    |      &#9745;      |      &#9745;      |
|                   [Adaptive Patch](https://openreview.net/pdf?id=_wSHsgrVali)                    |    Poison    |    &#9745;    |      &#9745;      |      &#9745;      |
| [Latent Backdoor](https://people.cs.uchicago.edu/~ravenben/publications/pdf/pbackdoor-ccs19.pdf) | Supply-Chain |    &#9745;    |      &#9745;      |      &#9745;      |
|                          [WaNet](https://arxiv.org/pdf/2102.10369.pdf)                           |    Poison    |    &#9744;    |      &#9745;      |      &#9745;      |
|                               Parameter-Controlled Backdoor (Ours)                               | Supply-Chain |    &#9745;    |      &#9745;      |      &#9745;      |
|                                Trigger-Scattered backdoor (Ours)                                 |    Poison    |    &#9745;    |      &#9745;      |      &#9745;      |

|                                     Defense                                     | Training Data | Inference-Time | CIFAR-10 | ImageNet |
|:-------------------------------------------------------------------------------:|:-------------:|:--------------:|:--------:|:--------:|
|            [Randomized Smoothing](https://arxiv.org/abs/1902.02918)             |    &#9744;    |    &#9745;     | &#9745;  | &#9745;  |
|                              Pivotal Tuning (Ours)                              |    &#9745;    |    &#9744;     | &#9745;  | &#9745;  |
|        [Neural Attention Distillation](https://arxiv.org/abs/2101.05930)        |    &#9745;    |    &#9744;     | &#9745;  | &#9745;  |
|                                   Fine-Tuning                                   |    &#9745;    |    &#9744;     | &#9745;  | &#9745;  |
|                [Fine-Pruning](https://arxiv.org/abs/1805.12185)                 |    &#9745;    |    &#9744;     | &#9745;  | &#9745;  |
| [Neural Cleanse](https://ieeexplore.ieee.org/iel7/8826229/8835208/08835365.pdf) |    &#9745;    |    &#9744;     | &#9745;  | &#9745;  |
|               [Shrink Pad](https://arxiv.org/pdf/2104.02361.pdf)                |    &#9744;    |    &#9745;     | &#9745;  | &#9745;  |
|          [Adversarial Training](https://arxiv.org/pdf/1706.06083.pdf)           |    &#9745;    |    &#9744;     | &#9745;  | &#9745;  |



## Supported Model Architectures
ResNet-18, ResNet-34 ResNet-50, openai/clip-vit-base-patch32, openai/clip-vit-large-patch14,
google/vit-base-patch16-224, google/vit-base-patch16-224-in21k

## Pre-Trained Backdoored Checkpoints 
**Blinded during review.**

## Additional Resources
**Blinded during review.**

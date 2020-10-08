# GS-WGAN
[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/mnemonics/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)](https://www.python.org/)

![](teaser.png)

This repository contains the implementation for [GS-WGAN: A Gradient-Sanitized Approach for Learning Differentially Private Generators (NeurIPS 2020)](https://arxiv.org/abs/2006.08265).

Contact: Dingfan Chen (dingfan.chen@cispa.saarland)

## Requirements 
The environment can be set up using [Anaconda](https://www.anaconda.com/download/) with the following commands:

``` setup
conda create --name gswgan-pytorch python=3.6
conda activate gswgan-pytorch
conda install pytorch=1.2.0 
conda install torchvision -c pytorch
pip install -r requirements.txt
```

## Training
#### Step 1. To warm-start the discriminators:
```warm-start
cd source
sh pretrain.sh
```
- To run the training in parallel: adjust the 
'meta_start' argument and run the script multiple times in parallel.
- Alternatively, you can download the pre-trained models using the links [below](#pre-trained-models). 
   
#### Step 2. To train the differentially private generator:
```train
cd source
python main.py -name 'ResNet_default' -ldir 'results/mnist/pretrain/ResNet_default'
```

Note: 
The default setting require ~22G GPU memory. 

## Evaluation
- To compute the privacy cost:
    ```privacy 
    cd evaluation
    python privacy_analysis.py -name 'ResNet_default'
    ```

## Pre-trained Models
Pre-trained model checkpoints can be downloaded using the links below.

|   |Generator  | Discriminators |  
|---|---|---|
|MNIST (DCGAN)| [link]() | [link]() | 
|MNIST (ResNet)| [link]() | [link]() | 
|Fashion-MNIST (DCGAN) | [link]() | [link]() | 
|Fashion-MNIST (ResNet) | [link]() | [link]() | 


## Citation
```bibtex
@inproceedings{neurips20chen,
title = {GS-WGAN: A Gradient-Sanitized Approach for Learning Differentially Private Generators},
author = {Dingfan Chen and Tribhuvanesh Orekondy and Mario Fritz},
year = {2020},
date = {2020-12-06},
booktitle = {Neural Information Processing Systems (NeurIPS)},
pubstate = {published},
tppubtype = {inproceedings}
}
```

## Acknowledgements

Our implementation uses the source code from the following repositories:

* [Improved Training of Wasserstein GANs (Pytorch)](https://github.com/caogang/wgan-gp.git)

* [Improved Training of Wasserstein GANs (Tensorflow)](https://github.com/igul222/improved_wgan_training)

* [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://github.com/ajbrock/BigGAN-PyTorch.git)

* [Progressive Growing of GANs](https://github.com/tkarras/progressive_growing_of_gans.git)

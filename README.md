# eva8_source
This is the source repo which contains reusable moduler components which will be pulled in every assignment EVA assignment starting from 7th assignment onwards.

## Introduction

### Objective
Objective of this repository is to contain all the required code in the form of moduler component which can be git cloned in respective assignment running in google colab.
<br>
This compoents are but not limited to:
- components and modules to download torchvision dataset, apply respective augmentation and perform required preprocessing like normalization.
- custom model architecture.
- training and testing code as custom module.
- utility function to plot loss-accuracy graph, misclassified images, GradCAM of images.

### Repository setup
Since all the essential modules are written in .py files which will be `git cloned` into main notebook, it is necessary to understand the structure of the repository.
Below is a quick look on how the repository is setup:
<br>
```
eva8_source/
  |
  |
  ├── README.md                           <- The top-level README for developers using this project.
  |
  ├── LICENSE                             <- Standered apache 2.0 based license
  |
  ├── dataset/
  │   ├── data.py                           <- Python file to download, process and create dataset for training. As well as required DataLoader class
  │   ├── transform_albumentation.py        <- We define test and train transformations/augmentation in this file using albumentation library.
  │
  ├── main/                   <- Directory to store code releated to training and evaluation loop
  │   ├── train.py            <- Python file where training code is defined. Forward and backward pass will be done by this.
  │   ├── eval.py             <- File to perform evaluation while training the model. We save misclassified images in last epoch.
  │
  ├── models/                <- Directory to store python file containing model architecture
  │   └── resnet.py          <- Sample python file containing resent18 and resnet32 architecture.
  |
  ├── util/                   <- Folder which contains python files containing utility function.
  │   ├── get_gradcam.py            <- Python file where training code is defined. Forward and backward pass will be done by this.
  │   ├── plot_graph.py             <- File to perform evaluation while training the model. It on performs forward pass with no gradient calc.
  │   ├── plot_misclassified.p      <- File to perform evaluation while training the model. It on performs forward pass with no gradient calc.
```

### Getting started
To start using this repo you can follow [Assignment-7](https://github.com/devdastl/EVA-8_Phase-1_Assignment-7/blob/main/eva8_assignment_7.ipynb) notebook.
 - we clone this repository inside the colab runtime and setup correct working directory.
 - `import` required class, modules and function as mentioned in above colab notebook.
 - setup required optimization, this structure can change in future to incorporate functionality like One-Cycle-Policy to train model faster.





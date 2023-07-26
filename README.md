# A Comparison of Model Architectures for Missing Pixel Predictions

Pittawat TAVEEKITWORACHAI - 66132200174

Email: gr0609fv@ed.ritsumei.ac.jp

## Table of Contents
- [A Comparison of Model Architectures for Missing Pixel Predictions](#a-comparison-of-model-architectures-for-missing-pixel-predictions)
  - [1. Abstract](#1-abstract)
  - [2. Installation](#2-installation)
  - [3. Usage](#3-usage)
  - [4. Models](#4-models)
  - [5. Datasets](#5-datasets)


## 1. Abstract

This study compares and analyzes model architectures for detecting missing pixels in images. The effectiveness of three models—the multi-layer perceptron (MLP), convolutional neural network (CNN), and MobileNetV2—is assessed, with separate training on two datasets. The first dataset consists of curated images provided by an instructor, while the second dataset is a 1K sampled subset of Tiny-ImageNet-2000. To evaluate the scalability of the MLP, a larger architecture is trained on the 1K-Tiny-ImageNet-2000 dataset. Additionally, the original MLP is trained on the entire Tiny-ImageNet-2000. Moreover, an open-sourced Stable Diffusion (SD) model, tailored for inpainting, is also investigated for a comprehensive comparison. The performance metric used is mean squared error. The results reveal that models developed using the larger dataset perform worse than those trained on the smaller dataset. In both datasets, the CNN and MobileNetV2, which are CNN-based architectures, consistently outperform the MLP, showcasing their efficiency in extracting useful features through convolutional layers. Unexpectedly, scaling the MLP to a bigger architecture does not enhance its performance. Contrary to general observations, where it excels in general inpainting tasks, the SD model exhibits inferior performance in this specific task. To improve performance for this task, it is recommended to consider diversifying datasets, exploring novel model architectures, and incorporating data augmentation and transfer learning approaches.

> Full paper is available in `paper.pdf` file.

## 2. Installation

To get started with the project, follow these steps:

1. Download this repository to your local machine:

2. Set up a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

> You may need to install required dependencies if  you encounter any errors.

## 3. Usage

This project provides multiple notebooks file which used to train and evaluate models. The following table provides a brief overview of each notebook.

| Notebook | Description |
| --- | --- |
| `mlp.ipynb` | Train a MLP model on an original dataset existed in `missing_pixels` folder |
| `cnn.ipynb` | Train a CNN model on an original dataset existed in `missing_pixels` folder |
| `mobilenetv2.ipynb` | Train a MobileNetV2 model on an original dataset existed in `missing_pixels` folder |
| `mlp.imagenet.ipynb` | Train a MLP model on a 1K sampled subset of Tiny-ImageNet-2000 dataset |
| `cnn.imagenet.ipynb` | Train a CNN model on a 1K sampled subset of Tiny-ImageNet-2000 dataset |
| `mobilenetv2.imagenet.ipynb` | Train a MobileNetV2 model on a 1K sampled subset of Tiny-ImageNet-2000 dataset |
| `mlp.large.imagenet.ipynb` | Train a larger MLP model on a 1K sampled subset of Tiny-ImageNet-2000 dataset |
| `mlp.imagenet.full.ipynb` | Train a MLP model on the full Tiny-ImageNet-2000 dataset |
| `stable_diffusion.ipynb` | Use a [Stable Diffusion model](https://huggingface.co/runwayml/stable-diffusion-inpainting) to generate a prediction |
| `evals.ipynb` | Evaluate the performance of all models |

Additional `utils.py` file contains helper functions used in the notebooks and is required to run the notebooks. Therefore, to run notebooks on Google Colab, you need to upload the `utils.py` file to the Colab environment.

`images/` contains input and output files of the Stable Diffusion model and output files from evaluations.

## 4. Models

Trained models are available in the `models/` directory. The following table provides a brief overview of each model.

| Model | Description |
| --- | --- |
| `model.mlp.h5` | A MLP model trained on an original dataset existed in `missing_pixels` folder |
| `model.cnn.h5` | A CNN model trained on an original dataset existed in `missing_pixels` folder |
| `model.mobilenet.h5` | A MobileNetV2 model trained on an original dataset existed in `missing_pixels` folder |
| `model.mlp.imagenet.h5` | A MLP model trained on a 1K sampled subset of Tiny-ImageNet-2000 dataset |
| `model.cnn.imagenet.h5` | A CNN model trained on a 1K sampled subset of Tiny-ImageNet-2000 dataset |
| `model.mobilenet.imagenet.h5` | A MobileNetV2 model trained on a 1K sampled subset of Tiny-ImageNet-2000 dataset |
| `model.mlp.large.imagenet.h5` | A larger MLP model trained on a 1K sampled subset of Tiny-ImageNet-2000 dataset |
| `model.mlp.full.imagenet.h5` | A MLP model trained on the full Tiny-ImageNet-2000 dataset |

These models are required to run `evals.ipynb` notebook. To run the notebook on Google Colab, you need to upload the models to the Colab environment.

## 5. Datasets

Datasets are available in the `missing_pixels/` and `datasets/` directories. The following table provides a brief overview of each dataset.

| Dataset | Description |
| --- | --- |
| `missing_pixels/` | An original dataset provided by an instructor |
| `datasets/` | A Tiny-ImageNet-2000 dataset |

Please note that you may need to unzip the `datasets.zip` file to get the full Tiny-ImageNet-2000 dataset. This dataset is required to run `<model>.imagenet.ipynb` notebooks. To run the notebooks on Google Colab, you need to upload the datasets to the Colab environment.

# ImageRecognitionDataset

Caltech101/256, CIFAR-10/100, MNIST/FashionMNIST, omniglot

## Requirement

- Python >= 3.7
- Poetry >= 1.2

## Install

### pip

```bash
pip install numpy pillow tqdm
```

### poetry

```bash
poetry install
```

## Usage

```bash
# Dataset Download 
python src/download.py --dataset {CIFAR10 | CIFAR100 | MNIST | fashionMNIST | caltech101 | caltech256 | omniglot}
# Calculate Dataset Mean Std
python src/calculate.py --dataset {CIFAR10 | CIFAR100 | MNIST | fashionMNIST | caltech101 | caltech256 | omniglot}
```

## Caluculated Result

### GrayScale dataset

|           dataset           |  mean  |   std  |
| :-------------------------: | :----: | :----: |
|         MNIST(train)        | 0.1307 | 0.3013 |
|     fashionMNIST(train)     | 0.2860 | 0.3202 |
| Omniglot(images_background) | 0.9221 | 0.2622 |

### RGB dataset

|         dataset        |       mean(R, G, B)      |       std(R, G, B)       |
| :--------------------: | :----------------------: | :----------------------: |
|     CIFAR10(train)     | (0.4914, 0.4822, 0.4465) | (0.2022, 0.1993, 0.2009) |
|     CIFAR100(train)    | (0.5071, 0.4865, 0.4409) | (0.2008, 0.1983, 0.2022) |
| Caltech101(all images) | (0.5487, 0.5313, 0.5050) | (0.2497, 0.2467, 0.2483) |
| Caltech256(all images) | (0.5520, 0.5336, 0.5050) | (0.2420, 0.2412, 0.2438) |

## Link

Mean and std calculations are based on　<https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/5>

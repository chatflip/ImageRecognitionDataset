ImageRecognitionDataset
====

Caltech101/256, CIFAR-10/100, MNIST/FashionMNIST, omniglot

## Requirement
python 3.x

## Install
### pip
```bash
$ pip install numpy pillow
```
### poetry
```bash
$ poetry install
```

## Usage
```bash
# Dataset Download 
$ python download.py --dataset {CIFAR10 | CIFAR100 | MNIST | fashionMNIST | caltech101 | caltech256 | omniglot}
# Calculate Dataset Mean Std
$ python calculate.py 
```

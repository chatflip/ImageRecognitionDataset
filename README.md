ImageRecognitionDataset
====

Caltech101/256, CIFAR-10/100, MNIST/FashionMNIST, omniglot

## Requirement
python 3.x

## Install
### pip
```bash
$ pip install numpy pillow tqdm
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



<!-- 
## 1channel dataset
| dataset | mean | std |
|:-------:|:----:|:---:|
| MNIST(train) | 0.1307 | 0.3081 |
| FashionMNIST(train) | 0.2860 | 0.3530 |
| Omniglot(images_background) | 0.9220 | 0.2681 |

## 3channel dataset
| dataset | mean(R, G, B) | std(R, G, B) |
|:-------:|:-------------:|:------------:|
| CIFAR10(train) | (0.4914, 0.4822, 0.4465) | (0.2370, 0.2435, 0.2616) |
| CIFAR100(train) | (0.5071, 0.4865, 0.4409) | (0.673, 0.2564, 0.2762) |
| Caltech101(all images) | (0.5487, 0.5313, 0.5050) | (0.3205, 0.3152, 0.3273) |
| Caltech256(all images) | (, , ) | (, , ) |
| ilsvrc2012(train) | (0.4812, 0.4575, 0.4079) | (0.2832, 0.2761, 0.2895) |
| places365(train 256×256) | (0.4578, 0.4414, 0.4078) | (0.2692, 0.2670, 0.2851) |

# original ilsvrc
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
-->


## Link
Mean and std calculations are based on　https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/5

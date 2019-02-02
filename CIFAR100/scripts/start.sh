#!/bin/sh
mkdir -p csv
mkdir -p data

wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -zxvf cifar-100-python.tar.gz
python py/parser.py
rm -rf cifar-100-python
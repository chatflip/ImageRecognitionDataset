#!/bin/sh
mkdir -p csv
mkdir -p data
mkdir -p datasets

wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
tar -xvf 256_ObjectCategories.tar -C ./datasets/
python py/randomlist.py
python py/symblic.py
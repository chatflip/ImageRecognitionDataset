#!/bin/sh
wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
tar -zxvf 101_ObjectCategories.tar.gz -C ./datasets/
python py/make_randomlist_caltech101.py
python py/symblic_caltech101.py
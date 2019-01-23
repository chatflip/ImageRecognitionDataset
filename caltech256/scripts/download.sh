#!/bin/sh
wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
tar -xvf 256_ObjectCategories.tar -C ./datasets/
python py/make_randomlist_caltech256.py
python py/symblic_caltech256.py
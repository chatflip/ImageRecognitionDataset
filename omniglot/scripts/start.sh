#!/bin/sh
mkdir -p data
mkdir -p csv
wget https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip
wget https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip
unzip images_background.zip
unzip images_evaluation.zip
python py/parser.py
python py/randomlist.py
python py/symblic.py
rm -rf images_background
rm -rf images_evaluation

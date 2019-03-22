#!/bin/sh
wget https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip
wget https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip
python py/parser.py

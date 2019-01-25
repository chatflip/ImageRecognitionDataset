import os
import sys
import random
import argparse
import numpy as np

if __name__ == '__main__':
    root = os.getcwd()
    src_root = root+"/datasets/101_ObjectCategories/"
    dst_root = root+"/data/"
    for num_subset in range(0,10):
        if not os.path.exists(dst_root+"subset"+str(num_subset)):
            os.makedirs(dst_root+"subset"+str(num_subset))
            os.makedirs(dst_root+"subset"+str(num_subset)+"/train")
            os.makedirs(dst_root+"subset"+str(num_subset)+"/test")
        class_names = os.listdir(src_root)
        class_names.sort()
        for class_name in class_names:
            if "BACKGROUND" in class_name:
                continue
            elif not os.path.exists(dst_root+"subset"+str(num_subset)+"/train/"+class_name):
                os.makedirs(dst_root+"subset"+str(num_subset)+"/train/"+class_name)
                os.makedirs(dst_root+"subset"+str(num_subset)+"/test/"+class_name)
        train_lists = np.genfromtxt(root+"/csv/caltech101_train_subset"+str(num_subset)+".csv",dtype=np.str)
        test_lists = np.genfromtxt(root+"/csv/caltech101_test_subset"+str(num_subset)+".csv",dtype=np.str)
        dst_subset_root = dst_root+"subset"+str(num_subset)+"/"
        for train_list in train_lists:
            os.symlink(src_root+train_list, dst_subset_root+"train/"+train_list)
        for test_list in test_lists:
            os.symlink(src_root+test_list, dst_subset_root+"test/"+test_list)

import os
import random

import numpy as np

if __name__ == '__main__':
    root = os.getcwd()
    src_root = root+"/data/original/"
    dst_root = root+"/data/"
    for num_subset in range(20):
        if not os.path.exists(dst_root+"subset"+str(num_subset)):
            os.makedirs(dst_root+"subset"+str(num_subset))
            os.makedirs(dst_root+"subset"+str(num_subset)+"/train")
            os.makedirs(dst_root+"subset"+str(num_subset)+"/test")
        class_names = os.listdir(src_root)
        class_names.sort()
        for class_name in class_names:
            if not os.path.exists(dst_root+"subset"+str(num_subset)+"/train/"+class_name):
                os.makedirs(dst_root+"subset"+str(num_subset)+"/train/"+class_name)
                os.makedirs(dst_root+"subset"+str(num_subset)+"/test/"+class_name)
            file_names = os.listdir(src_root+class_name+"/")
            dst_subset_root = dst_root+"subset"+str(num_subset)+"/"
            for file_name in file_names:
                if "{:02d}.png".format(num_subset+1) in file_name:
                    os.symlink(src_root+class_name+"/"+file_name, dst_subset_root+"train/"+class_name+"/"+file_name)
                else:
                    os.symlink(src_root+class_name+"/"+file_name, dst_subset_root+"test/"+class_name+"/"+file_name)

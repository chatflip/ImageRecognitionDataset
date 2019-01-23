import os
import sys
import random
import argparse
import numpy as np

if __name__ == '__main__':
    random.seed(0)
    src_root = "./datasets/101_ObjectCategories/"
    class_names = os.listdir(src_root)
    class_names.sort()
    for num_subset in range(0,10):
        train_list = []
        test_list = []
        for class_name in class_names:
            if "BACKGROUND" in class_name:
                continue
            file_names = os.listdir(src_root+class_name+"/")

            file_names.sort()
            random.shuffle(file_names)
            train_count = 0
            for file_name in file_names:
                if train_count < 30:
                    train_list.append(class_name+"/"+file_name)
                elif train_count < 60:
                    test_list.append(class_name+"/"+file_name)
                train_count += 1
        np.savetxt("./csv/caltech101_train_subset"+str(num_subset)+".csv", train_list, fmt="%s")
        np.savetxt("./csv/caltech101_test_subset"+str(num_subset)+".csv", test_list, fmt="%s")
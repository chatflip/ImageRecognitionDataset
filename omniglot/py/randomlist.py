import os
import random

import numpy as np

if __name__ == '__main__':
    src_root = "data/original/"
    dst_root = "data/"
    class_names = os.listdir(src_root)
    class_names.sort()
    test_num = 20//5
    for num_subset in range(5):
        train_list = []
        test_list = []
        for class_name in class_names:
            file_names = os.listdir(src_root+class_name+"/")
            file_names.sort()
            for i, file_name in enumerate(file_names):
                if i in range(num_subset*test_num, num_subset*test_num+test_num):
                    test_list.append(class_name+"/"+file_name)
                else:
                    train_list.append(class_name+"/"+file_name)
        train_list.sort()
        test_list.sort()
        np.savetxt("./csv/omniglot_train_subset{}.csv".format(num_subset), train_list, fmt="%s")
        np.savetxt("./csv/omniglot_test_subset{}.csv".format(num_subset), test_list, fmt="%s")

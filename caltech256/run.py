import os
import tarfile
import shutil
import sys
import urllib.request

import numpy as np


def progress(block_count, block_size, total_size):
    percentage = min(int(100.0 * block_count * block_size / total_size), 100)
    bar = '[{}>{}]'.format('='*(percentage//4), ' '*(25-percentage//4))
    sys.stdout.write('{} {:3d}%\r'.format(bar, percentage))
    sys.stdout.flush()


def download(baseurl, filename):
    try:
        urllib.request.urlretrieve(url=baseurl+filename,
                                   filename=filename,
                                   reporthook=progress)
        print('')
    except (OSError, urllib.error.HTTPError) as err:
        print('ERROR :{}'.fromat(err.code))
        print(err.reason)


def untar(src):
    with tarfile.open(src, 'r:') as tr:
        tr.extractall('data/')


def make_subset(src):
    root = os.getcwd()
    src_root = '{}/data/{}/'.format(root, src)
    dst_root = '{}/data/'.format(root)
    for num_subset in range(10):
        subset_root = '{}subset{}/'.format(dst_root, num_subset)
        mkdir(subset_root)
        mkdir(subset_root+'train')
        mkdir(subset_root+'test')
        class_names = os.listdir(src_root)
        class_names.sort()
        for class_name in class_names:
            if "257.clutter" in class_name:
                continue
            mkdir('{}train/{}'.format(subset_root, class_name))
            mkdir('{}test/{}'.format(subset_root, class_name))
        train_lists = np.genfromtxt("csv/caltech256_train_subset{}.csv"
                                    .format(num_subset), dtype=np.str)
        test_lists = np.genfromtxt("csv/caltech256_test_subset{}.csv"
                                   .format(num_subset), dtype=np.str)
        for train_list in train_lists:
            os.symlink(src_root+train_list, subset_root+"train/"+train_list)
        for test_list in test_lists:
            os.symlink(src_root+test_list, subset_root+"test/"+test_list)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    mkdir('data')
    baseurl = 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/'
    tar_filename = '256_ObjectCategories.tar'
    print('Downloading: {}'.format(tar_filename))
    download(baseurl, tar_filename)
    untar(tar_filename)
    # Create subset
    print('Create subset: 256_ObjectCategories.tar')
    make_subset('256_ObjectCategories')

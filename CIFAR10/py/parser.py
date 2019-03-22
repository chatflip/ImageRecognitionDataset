import os
import tarfile
import shutil

import numpy as np
from PIL import Image

try:
    import cPickle as pickle
except:
    import pickle


def untar(src):
    with tarfile.open(src, 'r:gz') as tr:
        tr.extractall()


def unpickle(file):
    with open(file, 'rb') as fo:
        try:
            dict = pickle.load(fo, encoding='latin1')
        except:
            dict = pickle.load(fo)
    fo.close()
    return dict


def conv_data2image(data):
    return np.rollaxis(data.reshape((3, 32, 32)), 0, 3)


def remove(target):
    shutil.rmtree(target)

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    tar_filename = 'cifar-10-python.tar.gz'
    folder_path = "cifar-10-batches-py"
    untar(tar_filename)
    meta_data = unpickle(folder_path+'/batches.meta')
    class_names = meta_data['label_names']
    if not os.path.exists('data/train'):
        os.makedirs('data/train')
        os.makedirs('data/test')
        for class_name in class_names:
            os.makedirs('data/train/'+class_name)
            os.makedirs('data/test/'+class_name)
    #np.savetxt('csv/class_name.csv', meta_data['label_names'], fmt='%s')

    for num_subset in range(1, 6):
        train = unpickle(folder_path+'/data_batch_'+str(num_subset))
        train_datas = train['data']
        train_labels = train['labels']
        train_filenames = train['filenames']
        for data, label, filename in zip(train_datas, train_labels, train_filenames):
            pilimg = Image.fromarray(np.uint8(conv_data2image(data)))
            pilimg.save('data/train/'+class_names[label]+'/'+filename)

    test = unpickle(folder_path+'/test_batch')
    test_datas = test['data']
    test_labels = test['labels']
    test_filenames = test['filenames']
    for data, label, filename in zip(test_datas, test_labels, test_filenames):
        pilimg = Image.fromarray(np.uint8(conv_data2image(data)))
        pilimg.save('data/test/'+class_names[label]+'/'+filename)
    print('Remove untar file')
    remove(folder_path)

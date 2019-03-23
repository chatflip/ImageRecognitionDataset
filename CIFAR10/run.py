import os
import tarfile
import shutil
import sys
import urllib.request

import numpy as np
from PIL import Image

try:
    import cPickle as pickle
except:
    import pickle


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
    baseurl = 'https://www.cs.toronto.edu/~kriz/'
    tar_filename = 'cifar-10-python.tar.gz'
    print('Downloading: {}'.format(tar_filename))
    download(baseurl, tar_filename)
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

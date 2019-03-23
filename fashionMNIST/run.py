import gzip
import os
import sys
import urllib.request

import numpy as np
from PIL import Image


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

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '{}-labels-idx1-ubyte.gz'.format(kind))
    images_path = os.path.join(path, '{}-images-idx3-ubyte.gz'.format(kind))
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    baseurl = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz',
             ]
    for file in files:
        print('Downloading: {}'.format(file))
        download(baseurl, file)
    train_images, train_labels = load_mnist("", kind='train')
    count = 0
    for image, label in zip(train_images, train_labels):
        if not os.path.exists("data/train/{}".format(label)):
            os.makedirs("data/train/{}".format(label))
        image = np.reshape(image, (28, 28)).astype(np.uint8)
        pilimg = Image.fromarray(image)
        pilimg.save("data/train/{0}/{1:05d}.png".format(label, count))
        count += 1

    test_images, test_labels = load_mnist("", kind='t10k')
    count = 0
    for image, label in zip(test_images, test_labels):
        if not os.path.exists("data/test/{}".format(label)):
            os.makedirs("data/test/{}".format(label))
        image = np.reshape(image, (28, 28)).astype(np.uint8)
        pilimg = Image.fromarray(image)
        pilimg.save("data/test/{0}/{1:05d}.png".format(label, count))
        count += 1

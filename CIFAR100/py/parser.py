import os
import cPickle
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
       dict = cPickle.load(fo)
    fo.close()
    return dict

def conv_data2image(data):
    return np.rollaxis(data.reshape((3,32,32)),0,3)

if __name__ == '__main__':
    folder_path = "cifar-100-python"
    meta_data = unpickle(folder_path+'/meta')
    class_names = meta_data['fine_label_names']
    if not os.path.exists('data/train'):
        os.makedirs('data/train')
        os.makedirs('data/test')
        for class_name in class_names:
            os.makedirs('data/train/'+class_name)
            os.makedirs('data/test/'+class_name)
    np.savetxt('csv/subclass_name.csv',meta_data['fine_label_names'],fmt='%s')

    train = unpickle(folder_path+'/train')
    train_datas = train['data']
    train_labels = train['fine_labels']
    train_filenames = train['filenames']
    subset_count = 0
    for data,label,filename in zip(train_datas,train_labels,train_filenames):
        pilimg = Image.fromarray(np.uint8(conv_data2image(data)))
        pilimg.save('data/train/'+class_names[label]+'/'+filename)

    test = unpickle(folder_path+'/test')
    test_datas = test['data']
    test_labels = test['fine_labels']
    test_filenames = test['filenames']
    for data,label,filename in zip(test_datas,test_labels,test_filenames):
        pilimg = Image.fromarray(np.uint8(conv_data2image(data)))
        pilimg.save('data/test/'+class_names[label]+'/'+filename)
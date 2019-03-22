import os
import random
import shutil
import zipfile


def unzip(target):
    with zipfile.ZipFile(target, 'r') as z:
        z.extractall()


def convert(src, dst):
    num2class = {}
    mkdir(dst)
    for root, dirs, file_names in os.walk(src):
        if len(dirs) == 0:
            tmp = file_names[0]
            class_num = int(tmp[:int(tmp.find('_'))])
            num2class.setdefault(class_num, root)

    for key, value in num2class.items():
        root, class_name, subclass_name = value.split('/')
        dst_path = '{}{:04d}_{}_{}'.format(dst, key, class_name, subclass_name)
        mkdir(dst_path)
        for file_name in os.listdir(value):
            shutil.copy(os.path.join(value, file_name),
                        os.path.join(dst_path, file_name))


def make_subset(src):
    root = os.getcwd()
    src_root = '{}/data/{}/'.format(root, src)
    dst_root = '{}/data/'.format(root)
    for num_subset in range(20):
        print('Create subset: {}/20'.format(num_subset+1))
        subset_root = '{}subset{}/{}/'.format(dst_root, num_subset, src)
        mkdir(subset_root)
        mkdir(subset_root+'train')
        mkdir(subset_root+'test')
        class_names = os.listdir(src_root)
        class_names.sort()
        for class_name in class_names:
            mkdir('{}train/{}'.format(subset_root, class_name))
            mkdir('{}test/{}'.format(subset_root, class_name))
            file_names = os.listdir(src_root+class_name+'/')
            for file_name in file_names:
                if '{:02d}.png'.format(num_subset+1) in file_name:
                    os.symlink(src_root+class_name+'/'+file_name,
                               subset_root+'train/'+class_name+'/'+file_name)
                else:
                    os.symlink(src_root+class_name+'/'+file_name,
                               subset_root+'test/'+class_name+'/'+file_name)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove(target):
    filename, _ = os.path.splitext(target)
    shutil.rmtree(filename)

if __name__ == '__main__':
    mkdir('data')
    zipfile1 = 'images_background.zip'
    zipfile2 = 'images_evaluation.zip'
    # Extract unzip files
    print('Extract zipfile: {}'.format(zipfile1))
    unzip(zipfile1)
    print('Extract zipfile: {}'.format(zipfile2))
    unzip(zipfile2)
    # Rearranging category order
    src_root = 'images_background/'
    dst_root = 'data/' + src_root
    src_root2 = 'images_evaluation/'
    dst_root2 = 'data/' + src_root2
    print('Rearranging category order')
    convert(src_root, dst_root)
    convert(src_root2, dst_root2)
    # Create subset
    print('Create subset')
    make_subset('images_background')
    make_subset('images_evaluation')
    # Remove unzip files
    print('Remove unzipfile')
    remove(zipfile1)
    remove(zipfile2)

import os
import shutil
import sys
import urllib.request
import zipfile


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
    baseurl = 'https://raw.githubusercontent.com/'\
              'brendenlake/omniglot/master/python/'
    zipfile1 = 'images_background.zip'
    zipfile2 = 'images_evaluation.zip'
    src_root1 = 'images_background/'
    dst_root1 = 'data/' + src_root1
    src_root2 = 'images_evaluation/'
    dst_root2 = 'data/' + src_root2
    # Download zipfiles
    print('Downloading: {}'.format(zipfile1))
    download(baseurl, zipfile1)
    print('Downloading: {}'.format(zipfile2))
    download(baseurl, zipfile2)
    # Extract unzip files
    print('Extract zipfile: {}'.format(zipfile1))
    unzip(zipfile1)
    print('Extract zipfile: {}'.format(zipfile2))
    unzip(zipfile2)
    # Rearranging category order
    print('Rearranging category order')
    convert(src_root1, dst_root1)
    convert(src_root2, dst_root2)
    # Create subset
    print('Create subset: images_background')
    make_subset('images_background')
    print('Create subset: images_evaluation')
    make_subset('images_evaluation')
    # Remove unzip files
    print('Remove unzip file')
    remove(zipfile1)
    remove(zipfile2)

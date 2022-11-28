import gzip
import os
import pickle
import shutil
import sys
import tarfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image


class ExpansionDataset(object):
    """docstring for ClassName"""

    def __init__(self, dataset_name, raw_path, data_path):
        print(dataset_name)
        self.dataset_name = dataset_name
        self.raw_path = os.path.expanduser(raw_path)

        self.data_path = os.path.expanduser(data_path)
        self.download_dict = get_url(self.dataset_name)

    def download(self):
        for filename in [*self.download_dict["filenames"]]:
            download_file(self.download_dict["baseurl"], filename, self.raw_path)

    def decompress(self):
        print("Decompress: {}".format(self.dataset_name))
        for filename in [*self.download_dict["filenames"]]:
            decompress_file(filename, self.raw_path)

    def setup(self):
        setup_file(self.dataset_name, self.raw_path, self.data_path)


def get_url(dataset_name):
    data = {
        "CIFAR10": {
            "baseurl": "https://www.cs.toronto.edu/~kriz/",
            "filenames": ["cifar-10-python.tar.gz"],
        },
        "CIFAR100": {
            "baseurl": "https://www.cs.toronto.edu/~kriz/",
            "filenames": ["cifar-100-python.tar.gz"],
        },
        "MNIST": {
            "baseurl": "http://yann.lecun.com/exdb/mnist/",
            "filenames": [
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            ],
        },
        "fashionMNIST": {
            "baseurl": "http://fashion-mnist.s3-website." "eu-central-1.amazonaws.com/",
            "filenames": [
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            ],
        },
        "caltech101": {
            "baseurl": "http://www.vision.caltech.edu/" "Image_Datasets/Caltech101/",
            "filenames": ["101_ObjectCategories.tar.gz"],
        },
        "caltech256": {
            "baseurl": "http://www.vision.caltech.edu/" "Image_Datasets/Caltech256/",
            "filenames": ["256_ObjectCategories.tar"],
        },
        "omniglot": {
            "baseurl": "https://raw.githubusercontent.com/"
            "brendenlake/omniglot/master/python/",
            "filenames": ["images_background.zip", "images_evaluation.zip"],
        },
    }
    return data[dataset_name]


def progress(block_count, block_size, total_size):
    percentage = min(int(100.0 * block_count * block_size / total_size), 100)
    bar = "[{}>{}]".format("=" * (percentage // 4), " " * (25 - percentage // 4))
    sys.stdout.write("{} {:3d}%\r".format(bar, percentage))
    sys.stdout.flush()


def download_file(baseurl, filename, raw_path):
    if os.path.exists(os.path.join(raw_path, filename)):
        print("File exists: {}".format(filename))
    else:
        print("Downloading: {}".format(filename))
        try:
            urllib.request.urlretrieve(
                url=baseurl + filename,
                filename=os.path.join(raw_path, filename),
                reporthook=progress,
            )
            print("")
        except (OSError, urllib.error.HTTPError) as err:
            print("ERROR :{}".format(err.code))
            print(err.reason)


def decompress_file(filename, raw_path):
    if ".tar.gz" in filename:
        with tarfile.open(os.path.join(raw_path, filename), "r:gz") as tr:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tr, os.path.join(raw_path,""))
    elif ".tar" in filename:
        with tarfile.open(os.path.join(raw_path, filename), "r:") as tr:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tr, os.path.join(raw_path,""))
    elif ".zip" in filename:
        with zipfile.ZipFile(os.path.join(raw_path, filename), "r") as z:
            z.extractall(os.path.join(raw_path, ""))


def setup_file(dataset_name, raw_path, data_path):
    exist_mkdir(os.path.join(data_path, dataset_name))
    if dataset_name == "CIFAR10":
        setup_cifar10(dataset_name, raw_path, data_path)
    elif dataset_name == "CIFAR100":
        setup_cifar100(dataset_name, raw_path, data_path)
    elif dataset_name == "MNIST":
        setup_mnist(dataset_name, raw_path, data_path)
    elif dataset_name == "fashionMNIST":
        setup_fashionmnist(dataset_name, raw_path, data_path)
    elif dataset_name == "caltech101":
        setup_caltech101(dataset_name, raw_path, data_path)
    elif dataset_name == "caltech256":
        setup_caltech256(dataset_name, raw_path, data_path)
    elif dataset_name == "omniglot":
        setup_omniglot(dataset_name, raw_path, data_path)


def exist_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unpickle(file):
    with open(file, "rb") as f:
        pic = pickle.load(f, encoding="latin1")
    return pic


def data2img_cifar(dataset_name, src, dst, class_names):
    pickles = unpickle(src)
    datas = pickles["data"]
    if dataset_name == "CIFAR10":
        labels = pickles["labels"]
    elif dataset_name == "CIFAR100":
        labels = pickles["fine_labels"]
    filenames = pickles["filenames"]
    for data, label, filename in zip(datas, labels, filenames):
        img = np.rollaxis(data.reshape((3, 32, 32)), 0, 3)
        pilimg = Image.fromarray(np.uint8(img))
        pilimg.save(os.path.join(dst, class_names[label], filename))


def setup_cifar10(dataset_name, raw_path, data_path):
    folder_name = "cifar-10-batches-py"
    src_root = os.path.join(raw_path, folder_name)
    dst_root = os.path.join(data_path, dataset_name)
    meta_data = unpickle(os.path.join(src_root, "batches.meta"))
    class_names = meta_data["label_names"]
    exist_mkdir(os.path.join(dst_root, "train"))
    exist_mkdir(os.path.join(dst_root, "test"))
    for class_name in class_names:
        exist_mkdir(os.path.join(dst_root, "train", class_name))
        exist_mkdir(os.path.join(dst_root, "test", class_name))
    # Extract train files
    print("Extract train files")
    for num_subset in range(1, 6):
        src_path = "{}/data_batch_{}".format(src_root, num_subset)
        dst_path = os.path.join(dst_root, "train")
        data2img_cifar(dataset_name, src_path, dst_path, class_names)
    # Extract test files
    print("Extract test files")
    src_path = "{}/test_batch".format(src_root)
    dst_path = os.path.join(dst_root, "test")
    data2img_cifar(dataset_name, src_path, dst_path, class_names)


def setup_cifar100(dataset_name, raw_path, data_path):
    folder_name = "cifar-100-python"
    src_root = os.path.join(raw_path, folder_name)
    dst_root = os.path.join(data_path, dataset_name)
    meta_data = unpickle(os.path.join(src_root, "meta"))
    class_names = meta_data["fine_label_names"]
    exist_mkdir(os.path.join(data_path, dataset_name, "train"))
    exist_mkdir(os.path.join(data_path, dataset_name, "test"))
    for class_name in class_names:
        exist_mkdir(os.path.join(data_path, dataset_name, "train", class_name))
        exist_mkdir(os.path.join(data_path, dataset_name, "test", class_name))
    # Extract train files
    print("Extract train files")
    src_path = "{}/train".format(src_root)
    dst_path = os.path.join(dst_root, "train")
    data2img_cifar(dataset_name, src_path, dst_path, class_names)
    # Extract test files
    print("Extract test files")
    src_path = "{}/test".format(src_root)
    dst_path = os.path.join(dst_root, "test")
    data2img_cifar(dataset_name, src_path, dst_path, class_names)


def data2img_mnist(src, dst, phase):
    if phase == "train":
        prefix = "train"
    elif phase == "test":
        prefix = "t10k"
    imgs_path = os.path.join(src, "{}-images-idx3-ubyte.gz".format(prefix))
    labels_path = os.path.join(src, "{}-labels-idx1-ubyte.gz".format(prefix))
    with gzip.open(imgs_path, "rb") as img, gzip.open(labels_path, "rb") as lb:
        labels = np.frombuffer(lb.read(), dtype=np.uint8, offset=8)
        imgs = np.frombuffer(img.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )
        count = 0
        for img, label in zip(imgs, labels):
            exist_mkdir(os.path.join(dst, phase, str(label)))
            img = np.reshape(img, (28, 28)).astype(np.uint8)
            pilimg = Image.fromarray(img)
            pilimg.save("{}/{}/{}/{:05d}.png".format(dst, phase, label, count))
            count += 1


def setup_mnist(dataset_name, raw_path, data_path):
    # Extract train files
    print("Extract train files")
    dst_root = os.path.join(data_path, dataset_name)
    data2img_mnist(os.path.join(raw_path, ""), dst_root, "train")
    # Extract test files
    print("Extract test files")
    data2img_mnist(os.path.join(raw_path, ""), dst_root, "test")


def setup_fashionmnist(dataset_name, raw_path, data_path):
    # Extract train files
    print("Extract train files")
    dst_root = os.path.join(data_path, dataset_name)
    data2img_mnist(os.path.join(raw_path, ""), dst_root, "train")
    # Extract test files
    print("Extract test files")
    data2img_mnist(os.path.join(raw_path, ""), dst_root, "test")


def symlink_caltech(dataset_name, data_path, folder_name):
    data_path = os.path.abspath(data_path)
    if dataset_name == "caltech101":
        ignore_class = "BACKGROUND_Google"
    elif dataset_name == "caltech256":
        ignore_class = "257.clutter"
    sym_root = "{}/{}".format(data_path, dataset_name)
    for num_subset in range(10):
        subset = "subset{}".format(num_subset)
        subset_root = os.path.join(sym_root, subset)
        exist_mkdir(subset_root)
        exist_mkdir(os.path.join(subset_root, "train"))
        exist_mkdir(os.path.join(subset_root, "test"))
        class_names = os.listdir(os.path.join(sym_root, folder_name))
        class_names.sort()
        for class_name in class_names:
            if ignore_class in class_name:
                continue
            exist_mkdir(os.path.join(subset_root, "train", class_name))
            exist_mkdir(os.path.join(subset_root, "test", class_name))
        for phase in ("train", "test"):
            filenames = np.genfromtxt(
                "{0}/csv/{0}_{1}_{2}.csv".format(dataset_name, phase, subset),
                dtype=np.str,
            )
            for fname in filenames:
                if not os.path.exists(os.path.join(subset_root, phase, fname)):
                    os.symlink(
                        os.path.join(sym_root, folder_name, fname),
                        os.path.join(subset_root, phase, fname),
                    )


def setup_caltech101(dataset_name, raw_path, data_path):
    folder_name = "101_ObjectCategories"
    # copy 101_ObjectCategories
    cp_src = os.path.join(raw_path, folder_name)
    cp_dst = os.path.join(data_path, dataset_name, folder_name)
    if not os.path.exists(cp_dst):
        shutil.copytree(cp_src, cp_dst)
    symlink_caltech(dataset_name, data_path, folder_name)


def setup_caltech256(dataset_name, raw_path, data_path):
    folder_name = "256_ObjectCategories"
    # copy 256_ObjectCategories
    cp_src = os.path.join(raw_path, folder_name)
    cp_dst = os.path.join(data_path, dataset_name, folder_name)
    if not os.path.exists(cp_dst):
        shutil.copytree(cp_src, cp_dst)
    symlink_caltech(dataset_name, data_path, folder_name)


def convert_omniglot(src, dst):
    num2class = {}
    exist_mkdir(dst)
    for root, dirs, file_names in os.walk(src):
        if len(dirs) == 0:
            tmp = file_names[0]
            class_num = int(tmp[: int(tmp.find("_"))])
            num2class.setdefault(class_num, root)
    for key, value in num2class.items():
        _, _, class_name, subclass_name = value.split("/")
        dst_path = "{}/{:04d}_{}_{}".format(dst, key, class_name, subclass_name)
        exist_mkdir(dst_path)
        for file_name in os.listdir(value):
            shutil.copy(
                os.path.join(value, file_name), os.path.join(dst_path, file_name)
            )


def symlink_omniglot(dst_path, folder_name):
    dst_path = os.path.abspath(dst_path)
    src_root = "{}/{}".format(dst_path, folder_name)
    dst_root = "{}/".format(dst_path)
    for num_subset in range(20):
        subset_root = "{}/subset{}/{}".format(dst_root, num_subset, folder_name)
        exist_mkdir(subset_root)
        exist_mkdir(os.path.join(subset_root, "train"))
        exist_mkdir(os.path.join(subset_root, "test"))
        class_names = os.listdir(src_root)
        class_names.sort()
        for class_name in class_names:
            exist_mkdir(os.path.join(subset_root, "train", class_name))
            exist_mkdir(os.path.join(subset_root, "test", class_name))
            file_names = os.listdir(os.path.join(src_root, class_name))
            for file_name in file_names:
                if "{:02d}.png".format(num_subset + 1) in file_name:
                    os.symlink(
                        os.path.join(src_root, class_name, file_name),
                        os.path.join(subset_root, "train", class_name, file_name),
                    )
                else:
                    os.symlink(
                        os.path.join(src_root, class_name, file_name),
                        os.path.join(subset_root, "test", class_name, file_name),
                    )


def setup_omniglot(dataset_name, raw_path, data_path):
    for folder_name in ("images_background", "images_evaluation"):
        src_path = os.path.join(raw_path, folder_name)
        dst_path = os.path.join(data_path, dataset_name)
        convert_omniglot(src_path, os.path.join(dst_path, folder_name))
        symlink_omniglot(dst_path, folder_name)


def caltech101_list():
    import random

    random.seed(0)
    src_root = "caltech101/101_ObjectCategories"
    class_names = os.listdir(src_root)
    class_names.sort()
    for num_subset in range(0, 10):
        train_list = []
        test_list = []
        for class_name in class_names:
            if "BACKGROUND" in class_name:
                continue
            file_names = os.listdir(os.path.join(src_root, class_name))
            file_names.sort()
            random.shuffle(file_names)
            train_count = 0
            for file_name in file_names:
                if not file_name.endswith(".jpg"):
                    continue
                if train_count < 30:
                    train_list.append(os.path.join(src_root, class_name, file_name))
                elif train_count < 60:
                    test_list.append(os.path.join(src_root, class_name, file_name))
                train_count += 1
        np.savetxt(
            "{0}/csv/{0}_train_subset{1}.csv".format("caltech101", num_subset),
            train_list,
            fmt="%s",
        )
        np.savetxt(
            "{0}/csv/{0}_test_subset{1}.csv".format("caltech101", num_subset),
            test_list,
            fmt="%s",
        )


def caltech256_list():
    import random

    random.seed(0)
    src_root = "caltech256/256_ObjectCategories"
    class_names = os.listdir(src_root)
    class_names.sort()
    for num_subset in range(0, 10):
        train_list = []
        test_list = []
        for class_name in class_names:
            if "257.clutter" in class_name:
                continue
            file_names = os.listdir(os.path.join(src_root, class_name))
            file_names.sort()
            random.shuffle(file_names)
            train_count = 0
            for file_name in file_names:
                if not file_name.endswith(".jpg"):
                    continue
                if train_count < 30:
                    train_list.append(class_name + "/" + file_name)
                elif train_count < 60:
                    test_list.append(class_name + "/" + file_name)
                train_count += 1
        np.savetxt(
            "{0}/csv/{0}_train_subset{1}.csv".format("caltech256", num_subset),
            train_list,
            fmt="%s",
        )
        np.savetxt(
            "{0}/csv/{0}_test_subset{1}.csv".format("caltech256", num_subset),
            test_list,
            fmt="%s",
        )

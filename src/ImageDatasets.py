from __future__ import annotations

import gzip
import os
import pickle
import random
import shutil
import ssl
import sys
import tarfile
import urllib.request
import zipfile
from typing import Any

import numpy as np
import yaml
from PIL import Image


class ExpansionDataset:
    def __init__(
        self, dataset_name: str, raw_path: str, data_path: str, config_path: str
    ) -> None:
        """Instantiate an ExpansionDataset object.

        Args:
            dataset_name (str): Name of the dataset
            raw_path (str): Raw data directory path
            data_path (str): Processed data directory path
            config_path (str): Path to the dataset configuration path
        """
        self.dataset_name = dataset_name
        self.raw_path = os.path.expanduser(raw_path)
        self.data_path = os.path.expanduser(data_path)
        with open(config_path) as cfg:
            data = yaml.safe_load(cfg)
        self.download_dict = data[self.dataset_name]

    def download(self) -> None:
        """Download files from the remote server."""
        for filename in [*self.download_dict["filenames"]]:
            download_file(self.download_dict["baseurl"], filename, self.raw_path)

    def decompress(self) -> None:
        """Decompress files in the raw data directory."""
        print("Decompress: {}".format(self.dataset_name))
        for filename in [*self.download_dict["filenames"]]:
            decompress_file(filename, self.raw_path)

    def setup(self) -> None:
        """Set up the processed data directory."""
        setup_file(self.dataset_name, self.raw_path, self.data_path)


def progress(block_count: int, block_size: int, total_size: int) -> None:
    """Report download progress.

    Args:
        block_count (int): The number of blocks transferred so far [int].
        block_size (int): The size of each block in bytes [int].
        total_size (int): The total size of the file in bytes [int].
    """
    percentage = min(int(100.0 * block_count * block_size / total_size), 100)
    bar = "[{}>{}]".format("=" * (percentage // 4), " " * (25 - percentage // 4))
    sys.stdout.write("{} {:3d}%\r".format(bar, percentage))
    sys.stdout.flush()


def download_file(baseurl: str, filename: str, raw_path: str) -> None:
    """Download a file from the remote server.

    Args:
        baseurl (str): The URL of the remote server
        filename (str): The filename to download
        raw_path (str): The directory to save the downloaded file
    """
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
        except urllib.error.URLError as e:
            print(f"OSError :{e.reason}")
            print("Retry with ssl._create_unverified_context")
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(
                url=baseurl + filename,
                filename=os.path.join(raw_path, filename),
                reporthook=progress,
            )


def decompress_file(filename: str, raw_path: str) -> None:
    """Decompress a file in the raw data directory.

    Args:
        filename (str): The name of the file to decompress
        raw_path (str): The path of the raw data directory
    """
    if ".tar.gz" in filename:
        with tarfile.open(os.path.join(raw_path, filename), "r:gz") as tr:
            tr.extractall(os.path.join(raw_path, ""))
    elif ".tar" in filename:
        with tarfile.open(os.path.join(raw_path, filename), "r:") as tr:
            tr.extractall(os.path.join(raw_path, ""))
    elif ".zip" in filename:
        with zipfile.ZipFile(os.path.join(raw_path, filename), "r") as z:
            z.extractall(os.path.join(raw_path, ""))


def setup_file(dataset_name: str, raw_path: str, data_path: str) -> None:
    """Create necessary directories and setup dataset for specified dataset_name.

    Args:
        dataset_name (str): Name of the dataset to be setup.
        raw_path (str): Path to the directory containing raw dataset files.
        data_path (str): Path to the directory where the dataset will be stored.
    """
    os.makedirs(os.path.join(data_path, dataset_name), exist_ok=True)
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


def unpickle(file: str) -> Any:
    """Read and unpickle the specified file.

    Args:
        file (str): Path to the file to be unpickled.

    Returns:
        Any: The unpickled object from the specified file.
    """
    with open(file, "rb") as f:
        pic = pickle.load(f, encoding="latin1")
    return pic


def data2img_cifar(
    dataset_name: str, src: str, dst: str, class_names: list[str]
) -> None:
    """Convert CIFAR dataset files to images and store in the specified directory.

    Args:
        dataset_name (str): Name of the dataset to be converted.
        src (str): Path to the directory containing the dataset files.
        dst (str): Path to the directory where the images will be stored.
        class_names (list[str]): List of class names for the dataset.
    """
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


def setup_cifar10(dataset_name: str, raw_path: str, data_path: str) -> None:
    """Setup the CIFAR-10 dataset.

    Args:
        dataset_name (str): Name of the dataset to be setup.
        raw_path (str): Path to the directory containing raw dataset files.
        data_path (str): Path to the directory where the dataset will be stored.
    """
    folder_name = "cifar-10-batches-py"
    src_root = os.path.join(raw_path, folder_name)
    dst_root = os.path.join(data_path, dataset_name)
    meta_data = unpickle(os.path.join(src_root, "batches.meta"))
    class_names = meta_data["label_names"]
    os.makedirs(os.path.join(dst_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, "test"), exist_ok=True)
    for class_name in class_names:
        os.makedirs(os.path.join(dst_root, "train", class_name), exist_ok=True)
        os.makedirs(os.path.join(dst_root, "test", class_name), exist_ok=True)
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


def setup_cifar100(dataset_name: str, raw_path: str, data_path: str) -> None:
    """Setup the CIFAR-100 dataset.

    Args:
        dataset_name (str): Name of the dataset to be setup.
        raw_path (str): Path to the directory containing raw dataset files.
        data_path (str): Path to the directory where the dataset will be stored.
    """
    folder_name = "cifar-100-python"
    src_root = os.path.join(raw_path, folder_name)
    dst_root = os.path.join(data_path, dataset_name)
    meta_data = unpickle(os.path.join(src_root, "meta"))
    class_names = meta_data["fine_label_names"]
    os.makedirs(os.path.join(data_path, dataset_name, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_path, dataset_name, "test"), exist_ok=True)
    for class_name in class_names:
        os.makedirs(
            os.path.join(data_path, dataset_name, "train", class_name), exist_ok=True
        )
        os.makedirs(
            os.path.join(data_path, dataset_name, "test", class_name), exist_ok=True
        )
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


def data2img_mnist(src: str, dst: str, phase: str) -> None:
    """Extracts image data from MNIST dataset and saves as PNG images in a directory tree.

    Args:
        src (str): The directory path of MNIST dataset.
        dst (str): The directory path to save extracted images.
        phase (str): The name of the dataset subset to extract, either "train" or "test".
    """
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
            os.makedirs(os.path.join(dst, phase, str(label)), exist_ok=True)
            img = np.reshape(img, (28, 28)).astype(np.uint8)  # type: ignore
            pilimg = Image.fromarray(img)
            pilimg.save("{}/{}/{}/{:05d}.png".format(dst, phase, label, count))
            count += 1


def setup_mnist(dataset_name: str, raw_path: str, data_path: str) -> None:
    """Sets up MNIST dataset by extracting the image data and saving it as PNG images.

    Args:
        dataset_name (str): The name of the dataset to set up, i.e. "mnist".
        raw_path (str): The directory path of the raw data.
        data_path (str): The directory path to save the processed data.
    """
    # Extract train files
    print("Extract train files")
    dst_root = os.path.join(data_path, dataset_name)
    data2img_mnist(os.path.join(raw_path, ""), dst_root, "train")
    # Extract test files
    print("Extract test files")
    data2img_mnist(os.path.join(raw_path, ""), dst_root, "test")


def setup_fashionmnist(dataset_name: str, raw_path: str, data_path: str) -> None:
    """Sets up Fashion-MNIST dataset by extracting the image data and saving it as PNG images.

    Args:
        dataset_name (str): The name of the dataset to set up, i.e. "fashionmnist".
        raw_path (str): The directory path of the raw data.
        data_path (str): The directory path to save the processed data.
    """
    # Extract train files
    print("Extract train files")
    dst_root = os.path.join(data_path, dataset_name)
    data2img_mnist(os.path.join(raw_path, ""), dst_root, "train")
    # Extract test files
    print("Extract test files")
    data2img_mnist(os.path.join(raw_path, ""), dst_root, "test")


def symlink_caltech(dataset_name: str, data_path: str, folder_name: str) -> None:
    """Creates symbolic links for Caltech dataset.

    Args:
        dataset_name (str): The name of the Caltech dataset to set up, either "caltech101" or "caltech256".
        data_path (str): The directory path to save the processed data.
        folder_name (str): The name of the folder containing the raw data.
    """
    data_path = os.path.abspath(data_path)
    if dataset_name == "caltech101":
        ignore_class = "BACKGROUND_Google"
    elif dataset_name == "caltech256":
        ignore_class = "257.clutter"
    sym_root = "{}/{}".format(data_path, dataset_name)
    for num_subset in range(10):
        subset = "subset{}".format(num_subset)
        subset_root = os.path.join(sym_root, subset)
        os.makedirs(os.path.join(subset_root, "train"), exist_ok=True)
        os.makedirs(os.path.join(subset_root, "test"), exist_ok=True)
        class_names = os.listdir(os.path.join(sym_root, folder_name))
        class_names.sort()
        for class_name in class_names:
            if ignore_class in class_name:
                continue
            os.makedirs(os.path.join(subset_root, "train", class_name), exist_ok=True)
            os.makedirs(os.path.join(subset_root, "test", class_name), exist_ok=True)
        for phase in ("train", "test"):
            filenames = np.genfromtxt(
                "{0}/csv/{0}_{1}_{2}.csv".format(dataset_name, phase, subset),
                dtype=str,
            )
            for fname in filenames:
                if not os.path.exists(os.path.join(subset_root, phase, fname)):
                    os.symlink(
                        os.path.join(sym_root, folder_name, fname),
                        os.path.join(subset_root, phase, fname),
                    )


def setup_caltech101(dataset_name: str, raw_path: str, data_path: str) -> None:
    """Sets up Caltech 101 dataset by copying the raw data and creating symbolic links.


    Args:
        dataset_name (str): The name of the Caltech dataset to set up, i.e. "caltech101".
        raw_path (str): The directory path of the raw data.
        data_path (str): The directory path to save the processed data.
    """
    folder_name = "101_ObjectCategories"
    # copy 101_ObjectCategories
    cp_src = os.path.join(raw_path, folder_name)
    cp_dst = os.path.join(data_path, dataset_name, folder_name)
    if not os.path.exists(cp_dst):
        shutil.copytree(cp_src, cp_dst)
    symlink_caltech(dataset_name, data_path, folder_name)


def setup_caltech256(dataset_name: str, raw_path: str, data_path: str) -> None:
    """Sets up Caltech 256 dataset by copying the raw data and creating symbolic links.

    Args:
        dataset_name (str): The name of the Caltech dataset to set up, i.e. "caltech256".
        raw_path (str): The directory path of the raw data.
        data_path (str): The directory path to save the processed data.
    """
    folder_name = "256_ObjectCategories"
    # copy 256_ObjectCategories
    cp_src = os.path.join(raw_path, folder_name)
    cp_dst = os.path.join(data_path, dataset_name, folder_name)
    if not os.path.exists(cp_dst):
        shutil.copytree(cp_src, cp_dst)
    symlink_caltech(dataset_name, data_path, folder_name)


def convert_omniglot(src: str, dst: str) -> None:
    """Converts Omniglot dataset to the format suitable for few-shot learning.

    Args:
        src (str): The directory path to the Omniglot dataset.
        dst (str): The directory path where the converted dataset will be saved.
    """
    num2class = {}  # type: ignore
    os.makedirs(dst, exist_ok=True)
    for root, dirs, file_names in os.walk(src):
        if len(dirs) == 0:
            tmp = file_names[0]
            class_num = int(tmp[: int(tmp.find("_"))])
            num2class.setdefault(class_num, root)
    for key, value in num2class.items():
        _, _, class_name, subclass_name = value.split("/")
        dst_path = "{}/{:04d}_{}_{}".format(dst, key, class_name, subclass_name)
        os.makedirs(dst_path, exist_ok=True)
        for file_name in os.listdir(value):
            shutil.copy(
                os.path.join(value, file_name), os.path.join(dst_path, file_name)
            )


def symlink_omniglot(dst_path: str, folder_name: str) -> None:
    """Creates symlinks for Omniglot dataset.

    Args:
        dst_path (str): The directory path where the Omniglot dataset is saved.
        folder_name (str): The name of the folder to be symlinked.
    """
    dst_path = os.path.abspath(dst_path)
    src_root = "{}/{}".format(dst_path, folder_name)
    dst_root = "{}/".format(dst_path)
    for num_subset in range(20):
        subset_root = "{}/subset{}/{}".format(dst_root, num_subset, folder_name)
        os.makedirs(os.path.join(subset_root, "train"), exist_ok=True)
        os.makedirs(os.path.join(subset_root, "test"), exist_ok=True)
        class_names = os.listdir(src_root)
        class_names.sort()
        for class_name in class_names:
            os.makedirs(os.path.join(subset_root, "train", class_name), exist_ok=True)
            os.makedirs(os.path.join(subset_root, "test", class_name), exist_ok=True)
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


def setup_omniglot(dataset_name: str, raw_path: str, data_path: str) -> None:
    """Prepares Omniglot dataset for few-shot learning by converting and symlinking.

    Args:
        dataset_name (str): The name of the dataset to be prepared.
        raw_path (str): The directory path where the original Omniglot dataset is stored.
        data_path (str): The directory path where the converted and symlinked dataset will be saved.
    """
    for folder_name in ("images_background", "images_evaluation"):
        src_path = os.path.join(raw_path, folder_name)
        dst_path = os.path.join(data_path, dataset_name)
        convert_omniglot(src_path, os.path.join(dst_path, folder_name))
        symlink_omniglot(dst_path, folder_name)


def caltech101_list() -> None:
    """
    Creates train and test split for Caltech-101 dataset.
    Splits the dataset into 10 subsets and saves a list of file paths for train and test splits
    of each subset in CSV format.
    """
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


def caltech256_list() -> None:
    """
    Creates train and test split for Caltech-256 dataset.
    Splits the dataset into 10 subsets and saves a list of file paths for train and test splits
    of each subset in CSV format.
    """
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

import os

import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class CalculateMeanStd:
    def __init__(self, dataset, data_path):

        self.dataset = dataset
        self.data_path = os.path.expanduser(data_path)

    def caluculate(self):
        image_paths = self.search_image_path()
        mean = 0
        std = 0
        for image_path in tqdm(image_paths):
            image = Image.open(image_path)
            width, height = image.size
            if image.mode == "RGB":
                channel = 3
            elif image.mode == "L" or image.mode == "1":
                channel = 1
            else:
                print(f"image.mode: {image.mode}")
                raise NotImplementedError
            data = np.asarray(image)
            data = data.reshape(height * width, channel)
            mean += data.mean(0)
            std += data.std(0)
        mean /= len(image_paths)
        std /= len(image_paths)
        print(self.dataset)
        if image.mode == "RGB":
            print(
                f"mean R: {mean[0]:5.2f} G: {mean[1]:5.2f} B: {mean[2]:5.2f}\t"
                f"std R: {std[0]:5.2f} G: {std[1]:5.2f} B: {std[2]:5.2f}\n"
                f"mean R: {mean[0]/255:5.4f} G: {mean[1]/255:5.4f} B: {mean[2]/255:5.4f}\t"
                f"std R: {std[0]/255:5.4f} G: {std[1]/255:5.4f} B: {std[2]/255:5.4f}"
            )
        elif image.mode == "L" or image.mode == "1":
            if image.mode == "1":
                mean *= 255
                std *= 255
            print(
                f"mean: {mean[0]:5.2f}\tstd: {std[0]:5.2f}\n"
                f"mean: {mean[0]/255:5.4f}\tstd : {std[0]/255:5.4f}"
            )

    def search_image_path(self):
        root = self.get_image_root()
        image_paths = []
        for classname in os.listdir(root):
            class_root = os.path.join(root, classname)
            for filename in os.listdir(class_root):
                image_path = os.path.join(class_root, filename)
                if filename.lower().endswith(IMG_EXTENSIONS):
                    image_paths.append(image_path)
        return image_paths

    def get_image_root(self):
        if (
            self.dataset == "CIFAR10"
            or self.dataset == "CIFAR100"
            or self.dataset == "MNIST"
            or self.dataset == "fashionMNIST"
        ):
            return os.path.join(self.data_path, self.dataset, "train")
        elif self.dataset == "caltech101":
            return os.path.join(self.data_path, self.dataset, "101_ObjectCategories")
        elif self.dataset == "caltech256":
            return os.path.join(self.data_path, self.dataset, "256_ObjectCategories")
        elif self.dataset == "omniglot":
            return os.path.join(self.data_path, self.dataset, "images_background")
        else:
            return self.data_path

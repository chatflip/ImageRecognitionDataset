import argparse

from ImageDatasets import ExpansionDataset


def conf():
    parser = argparse.ArgumentParser(description="Image Recognition Dataset")
    parser.add_argument("--dataset", default="CIFAR10", type=str, help="select dataset")
    parser.add_argument("--raw_file_path", default="", type=str, help="select dataset")
    parser.add_argument(
        "--data_file_path", default="data", type=str, help="select dataset"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = conf()
    assert (
        args.dataset == "CIFAR10"
        or args.dataset == "CIFAR100"
        or args.dataset == "MNIST"
        or args.dataset == "fashionMNIST"
        or args.dataset == "omniglot"
        or args.dataset == "caltech101"
        or args.dataset == "caltech256"
    ), "select CIFAR10/100, MNIST/fashionMNIST, omniglot, caltech101/256"
    if not args.raw_file_path:
        args.raw_file_path = args.dataset
    worker = ExpansionDataset(args.dataset, args.raw_file_path, args.data_file_path)
    # Download files
    worker.download()
    # Extract unzip files
    worker.decompress()
    # Setup
    worker.setup()

from args import conf
from ImageDatasets import ExpansionDataset


def main(args):
    if not args.raw_file_path:
        args.raw_file_path = args.dataset
    worker = ExpansionDataset(args.dataset, args.raw_file_path, args.data_file_path)
    # Download files
    worker.download()
    # Extract unzip files
    worker.decompress()
    # Setup
    worker.setup()


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
    main(args)

import argparse


def conf():
    parser = argparse.ArgumentParser(description="Image Recognition Dataset")
    parser.add_argument("--dataset", default="CIFAR10", type=str, help="select dataset")
    parser.add_argument("--raw_file_path", default="", type=str, help="select dataset")
    parser.add_argument(
        "--data_file_path", default="data", type=str, help="select dataset"
    )
    args = parser.parse_args()
    return args

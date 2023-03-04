import argparse


def conf() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Recognition Dataset")
    parser.add_argument("--dataset", default="CIFAR10", type=str)
    parser.add_argument("--raw_file_path", default="", type=str)
    parser.add_argument("--data_file_path", default="data", type=str)
    parser.add_argument("--config_path", default="config/datasets.yaml", type=str)
    args = parser.parse_args()
    return args

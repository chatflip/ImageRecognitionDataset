from args import conf

if __name__ == '__main__':
    args = conf()
    worker = CalculateMeanStd(args.dataset, args.data_file_path)
    # caluculate mean/std
    worker.caluculate()

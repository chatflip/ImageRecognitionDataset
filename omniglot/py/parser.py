import os
import pprint
import shutil


if __name__ == '__main__':
    src_root1 = "images_background/"
    src_root2 = "images_evaluation/"
    dst_root = "data/original/"
    num2class = {}
    if not os.path.exists(dst_root):
        os.mkdir(dst_root)

    for src_root in (src_root1, src_root2):
        for root, dirs, file_names in os.walk(src_root):
            if len(dirs) == 0:
                tmp = file_names[0]
                class_num = int(tmp[:int(tmp.find("_"))])
                num2class.setdefault(class_num, root)
    pprint.pprint(num2class)

    for key, value in num2class.items():
        root, class_name, subclass_name = value.split("/")
        dst_path = "{}{:04d}_{}_{}".format(dst_root, key, class_name, subclass_name)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        for file_name in os.listdir(value):
            shutil.copy(os.path.join(value, file_name), os.path.join(dst_path, file_name))

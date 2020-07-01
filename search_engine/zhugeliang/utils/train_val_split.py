from zhugeliang.utils.config import get_data_dir
import numpy as np
import os


def train_val_split(data_path=None, train_path=None, val_path=None, train_ratio=None):
    fw_train = open(train_path, 'w', encoding='utf-8')
    fw_val = open(val_path, 'w', encoding='utf-8')

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:

            """
            if len(line[:-1].split(' ')) < 7: # window_size*2+1
                continue
            """

            ratio = np.random.random()
            if ratio < train_ratio:
                fw_train.write(line)
            else:
                fw_val.write(line)

    fw_train.close()
    fw_val.close()


if __name__ == '__main__':
    data_dir = get_data_dir()
    data_path = os.path.join(data_dir, "book_text.txt")
    train_path = os.path.join(data_dir, "book_text_train.txt")
    val_path = os.path.join(data_dir, "book_text_val.txt")
    train_ratio = 0.8

    train_val_split(data_path=data_path,
                    train_path=train_path,
                    val_path=val_path,
                    train_ratio=train_ratio)

    print("Split done! train_path: %s" % train_path)
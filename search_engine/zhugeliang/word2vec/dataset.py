from zhugeliang.utils.config import get_data_dir
import os
import tensorflow as tf
from zhugeliang.word2vec.dictionary import load_dictionary


def data_generator(input_path=None, word_cnt_dict=None):
    while True:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line[:-1]


def get_dataset(input_path=None, word_cnt_dict=None):
    dataset = tf.data.Dataset.from_generator(
        data_generator(input_path=input_path, word_cnt_dict=word_cnt_dict),
    )

    print(dataset.take(2))
    return dataset


if __name__ == "__main__":
    train_path = os.path.join(get_data_dir(), "book_seg_text.txt")
    word_cnt_dict_path = os.path.join(get_data_dir(), "book_dict.pkl")

    word_cnt_dict = load_dictionary(dict_path=word_cnt_dict_path)

    get_dataset(input_path=train_path, word_cnt_dict=word_cnt_dict)


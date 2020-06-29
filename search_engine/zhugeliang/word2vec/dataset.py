from zhugeliang.utils.config import get_data_dir
import os
import tensorflow as tf
from zhugeliang.word2vec.dictionary import load_dictionary
from zhugeliang.word2vec.sample import Sampler


def data_generator(input_path=None, dict_dir=None):
    word_cnt_dict, word2id_dict, id2word_dict = load_dictionary(dict_dir=dict_dir)

    window_size = 3
    num_neg = 4

    sampler = Sampler(word_cnt_dict=word_cnt_dict, word2id_dict=word2id_dict)

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            buf = line[:-1].split(' ')

            if len(buf) < window_size * 2 + 1:
                continue

            index_buf = []
            for word in buf:
                if word not in word2id_dict:
                    continue
                index_buf.append(word2id_dict[word])

            if len(index_buf) < window_size * 2 + 1:
                continue

            #index_buf = [word2id_dict[word] for word in buf]

            # Simply remove region out of window_size, use (window_size, target, window_size)
            for i in range(window_size, len(index_buf) - window_size):
                target = [index_buf[i]]
                contexts = index_buf[i-window_size: i] + index_buf[i+1: i+1+window_size]
                negatives = sampler.sample(num_sample=num_neg, method="random")

                # [0] for label, positive always in the first position.
                yield (contexts, target, negatives), [0]


def train_generator():
    #input_path = os.path.join(get_data_dir(), "book_seg_text_train.txt")
    input_path = os.path.join(get_data_dir(), "shuf_train.txt")
    dict_dir = os.path.join(get_data_dir(), "book_dict")

    return data_generator(input_path=input_path, dict_dir=dict_dir)


def val_generator():
    #input_path = os.path.join(get_data_dir(), "book_seg_text_val.txt")
    input_path = os.path.join(get_data_dir(), "shuf_val.txt")
    dict_dir = os.path.join(get_data_dir(), "book_dict")

    return data_generator(input_path=input_path, dict_dir=dict_dir)


def get_dataset(generator, epochs=None, shuffle_buffer_size=None, batch_size=None):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_shapes=(((6, ), (1, ), (4, )), (1, )),
        output_types=((tf.int32, tf.int32, tf.int32), tf.int32)
    )

    return dataset.repeat(count=epochs)\
        .shuffle(buffer_size=shuffle_buffer_size)\
        .batch(batch_size=batch_size)


def get_train_dataset(epochs=None,
                      shuffle_buffer_size=None,
                      batch_size=None):
    return get_dataset(train_generator,
                       epochs=epochs,
                       shuffle_buffer_size=shuffle_buffer_size,
                       batch_size=batch_size)


def get_val_dataset(epochs=None,
                    shuffle_buffer_size=None,
                    batch_size=None):
    return get_dataset(val_generator,
                       epochs=epochs,
                       shuffle_buffer_size=shuffle_buffer_size,
                       batch_size=batch_size)


if __name__ == "__main__":
    train_path = os.path.join(get_data_dir(), "book_seg_text.txt")
    dict_dir = os.path.join(get_data_dir(), "book_dict")

    train_dataset = get_dataset(train_generator)
    print("===Train===")
    for data in train_dataset.take(2):
        print(data)

    val_dataset = get_dataset(val_generator)
    val_dataset = val_dataset.repeat().shuffle(buffer_size=10).batch(batch_size=4)
    print("\n===Val===")
    for data in val_dataset.take(2):
        print(data)


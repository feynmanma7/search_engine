from zhugeliang.utils.config import get_data_dir
import os
import tensorflow as tf
from zhugeliang.word2vec.dictionary import load_dictionary
from zhugeliang.word2vec.sample import Sampler
from zhugeliang.word2vec.word2vec import Word2vec


def get_dataset(input_path=None,
                dict_dir=None,
                epochs=10,
                shuffle_buffer_size=128,
                batch_size=8,
                window_size=3,
                num_neg=5):

    word_cnt_dict_path = os.path.join(dict_dir, "word_cnt_dict.pkl")
    word2id_dict_path = os.path.join(dict_dir, "word2id_dict.pkl")
    id2word_dict_path = os.path.join(dict_dir, "id2word_dict.pkl")

    word_cnt_dict = load_dictionary(dict_path=word_cnt_dict_path)
    word2id_dict = load_dictionary(dict_path=word2id_dict_path)
    id2word_dict = load_dictionary(dict_path=id2word_dict_path)

    sampler = Sampler(word_cnt_dict=word_cnt_dict, id2word_dict=id2word_dict)

    def generator():
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

                # Simply remove region out of window_size, use (window_size, target, window_size)
                for i in range(window_size, len(index_buf) - window_size):
                    target = [index_buf[i]]
                    contexts = index_buf[i - window_size: i] + index_buf[i + 1: i + 1 + window_size]
                    negatives = sampler.sample(num_sample=num_neg, method="weighted")

                    yield contexts, target, negatives

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_shapes=((window_size*2, ), (1, ), (num_neg, )),
        output_types=(tf.int32, tf.int32, tf.int32)
    )

    return dataset.repeat(count=epochs)\
        .shuffle(buffer_size=shuffle_buffer_size)\
        .batch(batch_size=batch_size)


if __name__ == "__main__":
    train_path = os.path.join(get_data_dir(), "ptb.train.txt")
    val_path = os.path.join(get_data_dir(), "ptb.valid.txt")

    dict_dir = os.path.join(get_data_dir(), "book_dict")

    """
    train_dataset = get_dataset(input_path=train_path,
                                dict_dir=dict_dir)
    print("===Train===")
    for data in train_dataset.take(2):
        print(data)
    """



    model = Word2vec()

    val_dataset = get_dataset(input_path=val_path,
                              dict_dir=dict_dir)
    print("\n===Val===")
    for contexts, target, negatives in val_dataset.take(2):
        print(contexts.shape, target.shape, negatives.shape)
        print(target)
        #loss = model(contexts, target, negatives)
        #print(loss, loss.shape)

    for contexts, target, negatives in val_dataset.take(2):
        print(contexts.shape, target.shape, negatives.shape)
        print(target)


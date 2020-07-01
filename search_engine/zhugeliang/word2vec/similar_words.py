from zhugeliang.utils.config import get_model_dir, get_data_dir
from zhugeliang.word2vec.dictionary import load_dictionary
from zhugeliang.word2vec.word2vec import Word2Vec
import tensorflow as tf
import os
import numpy as np


def get_word_representation(model=None, ):
    word_index = tf.constant([[0], [1], [2000], [300], [4]], dtype=tf.int32)
    print(word_index.shape)
    word_vec = model.get_last_layer_representation(word_index=word_index)
    print(word_vec.numpy())


def get_word_vectors(model=None, vocab_size=None, word_vec_path=None):
    fw = open(word_vec_path, 'w')

    batch_size = 8
    total_batch = vocab_size // batch_size + 1

    for batch in range(total_batch):
        word_index = tf.constant([[i] for i in range(batch*batch_size,
                                                     min((batch+1)*batch_size, vocab_size))],
                                 dtype=tf.int32)
        word_vec = model.get_last_layer_representation(word_index=word_index)
        word_vec = word_vec.numpy().tolist()

        for i, vec in enumerate(word_vec):
            index = batch * batch_size + i
            fw.write(str(index) + '\t' + ','.join(list(map(lambda x: '{:.4f}'.format(x), vec))) + '\n')

    fw.close()
    print("Write word vectors done! %s" % word_vec_path)


def find_most_similar_words(model=None, word_vec_path=None, word=None):
    # === Load dictionary
    dict_dir = os.path.join(get_data_dir(), "book_dict")

    word_cnt_dict_path = os.path.join(dict_dir, "word_cnt_dict.pkl")
    word_cnt_dict = load_dictionary(dict_path=word_cnt_dict_path)
    print(len(word_cnt_dict))

    word2id_dict_path = os.path.join(dict_dir, "word2id_dict.pkl")
    word2id_dict = load_dictionary(dict_path=word2id_dict_path)
    print(len(word2id_dict))

    id2word_dict_path = os.path.join(dict_dir, "id2word_dict.pkl")
    id2word_dict = load_dictionary(dict_path=id2word_dict_path)
    print(len(id2word_dict))

    if word not in word2id_dict:
        print('%s not in dict' % word)
        return None

    word_idx = word2id_dict[word]

    # === Load word_vec
    word_vecs = []
    with open(word_vec_path, 'r') as f:
        for line in f:
            buf = line[:-1].split('\t')
            vec = np.array(list(map(lambda x: float(x), buf[1].split(','))))
            word_vecs.append(vec)

    word_vecs = np.array(word_vecs)

    # === Find
    word_vec = word_vecs[word_idx]
    print('word_idx', word_idx)

    sims = np.dot(word_vecs, word_vec)
    ranks = np.argsort(-sims)
    print('ranks', ranks[:20])
    print('scores', sims[ranks[:20]])
    sim_words = [id2word_dict[idx] for idx in ranks[:20]]
    print("Top sim words of %s are: " % word)
    print(sim_words)


if __name__ == '__main__':
    checkpoint_dir = os.path.join(get_model_dir(), "word2vec")
    word_vec_path = os.path.join(get_data_dir(), "word_vectors")

    vocab_size = 30507
    window_size = 5
    num_neg = 8
    batch_size = 32

    # === Load model
    w2v = Word2Vec(vocab_size=vocab_size,
                   window_size=window_size,
                   num_neg=num_neg,
                   batch_size=batch_size)

    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    w2v.model.load_weights(checkpoint)

    get_word_vectors(model=w2v, vocab_size=vocab_size, word_vec_path=word_vec_path)

    word = 'optimization'
    find_most_similar_words(model=w2v, word_vec_path=word_vec_path, word=word)




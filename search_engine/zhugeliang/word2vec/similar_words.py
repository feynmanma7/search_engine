from zhugeliang.utils.config import get_model_dir, get_data_dir
from zhugeliang.word2vec.dictionary import load_dictionary
from zhugeliang.word2vec.word2vec import Word2vec, test_word2vec_once
import tensorflow as tf
import os
import numpy as np


def get_word_representation(model=None):
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

    vocab_size = 10001 # ptb, min_cnt = 5
    window_size = 5
    num_neg = 5
    embedding_dim = 64

    # === Load model
    checkpoint_dir = os.path.join(get_model_dir(), "word2vec")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    w2v = Word2vec(vocab_size=vocab_size,
                   window_size=window_size,
                   num_neg=num_neg,
                   embedding_dim=embedding_dim)

    optimizer = tf.keras.optimizers.Adam(0.001)
    w2v.compile(optimizer=optimizer)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=w2v)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    status = checkpoint.restore(latest)
    status.assert_existing_objects_matched()

    # === RUN ONCE!!! Important, must test the model once, then get the weights.
    test_word2vec_once(model=w2v)

    # === Get word vectors
    get_word_vectors(model=w2v, vocab_size=vocab_size, word_vec_path=word_vec_path)

    # === Find top sim words
    word = 'computer'
    find_most_similar_words(model=w2v, word_vec_path=word_vec_path, word=word)




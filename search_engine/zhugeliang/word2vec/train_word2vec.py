import tensorflow as tf
from zhugeliang.word2vec.word2vec import Word2Vec
from zhugeliang.utils.config import get_model_dir
import os


if __name__ == "__main__":
    batch_size = 2
    vocab_size = 10

    input_len = 5
    negative_len = 4

    inputs = tf.random.uniform((batch_size, input_len), minval=0, maxval=vocab_size, dtype=tf.int32)
    target = tf.random.uniform((batch_size, 1), minval=0, maxval=vocab_size, dtype=tf.int32)
    negatives = tf.random.uniform((batch_size, negative_len), minval=0, maxval=vocab_size, dtype=tf.int32)

    train_dataset = None
    val_dataset = None
    model_path = os.path.join(get_model_dir(), "word2vec")

    w2v = Word2Vec(vocab_size=vocab_size)
    w2v.train(train_dataset=train_dataset, val_dataset=val_dataset, model_path=model_path)

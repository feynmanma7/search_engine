import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras.layers import Embedding, Dense, Lambda, Input, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from zhugeliang.utils.metrics import norm
from zhugeliang.utils.config import get_logs_dir
from zhugeliang.word2vec.losses import dssm_loss
import os, datetime


class Word2vec(tf.keras.Model):
    def __init__(self,
                 vocab_size=10001,
                 window_size=3,
                 num_neg=5,
                 embedding_dim=32,
                 ):
        super(Word2vec, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = window_size
        self.num_neg = num_neg
        self.embedding_dim = embedding_dim

        self.input_embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_dim)
        self.output_embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_dim)

        
    def get_last_layer_representation(self, word_index=None):
        # word_index: [batch_size, 1], batch of word_index
        # word_vec: normed representation of word

        # word_embedding: [batch_size, 1, embedding_dim]
        word_embedding = self.output_embedding_layer(word_index)

        # word_embedding: [batch_size, embedding_dim]
        word_embedding = tf.squeeze(word_embedding, axis=1)

        # word_dense: [batch_size, dense_units]
        #word_dense = self.output_dense_layer(word_embedding)

        # === Normalize
        # word_norm: [batch_size, dense_units]
        word_norm = norm(word_embedding, normed_axis=1)

        return word_norm

    def call(self, contexts, target, negatives, training=None, mask=None):
        # === Embedding

        # [None, w*2, emb_dim]
        contexts_embedding = self.input_embedding_layer(contexts)

        # [None, 1, emb_dim]
        contexts = tf.reduce_mean(contexts_embedding, axis=1, keepdims=True)

        # [None, 1, emb_dim]
        target_embedding = self.output_embedding_layer(target)

        # [None, num_neg, emb_dim]
        negatives_embedding = self.output_embedding_layer(negatives)

        # === Cosine
        # [None, 1]
        target_cos = tf.keras.layers.Dot(axes=(2, 2), normalize=True)\
            ([target_embedding, contexts])

        # [None, num_neg]
        negatives_cos = tf.keras.layers.Dot(axes=(2, 2), normalize=True)\
            ([target_embedding, negatives_embedding])

        pos_loss = tf.reduce_mean(tf.sigmoid(target_cos))
        neg_loss = tf.reduce_mean(tf.sigmoid(-negatives_cos))

        #pos_loss = tf.reduce_mean(-tf.math.log(1 + tf.math.exp(-target_cos)))
        #neg_loss = tf.reduce_mean(-tf.math.log(1 + tf.math.exp(negatives_cos)))

        loss = pos_loss + neg_loss
        return loss



def test_word2vec_once(model=None):
    contexts = tf.random.uniform((8, 6), minval=0, maxval=10, dtype=tf.int32)
    target = tf.random.uniform((8, 1), minval=0, maxval=10, dtype=tf.int32)
    negatives = tf.random.uniform((8, 5), minval=0, maxval=10, dtype=tf.int32)

    loss = model(contexts, target, negatives)
    print(loss)
    #print("output_embedding_weights")
    #print(model.output_embedding_layer.get_weights())


if __name__ == '__main__':
    model = Word2vec()
    test_word2vec_once(model=model)
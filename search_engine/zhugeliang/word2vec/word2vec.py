import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras.layers import Embedding, Dense, Lambda, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from zhugeliang.utils.metrics import norm
from zhugeliang.utils.config import get_logs_dir
import os, datetime


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size=None, window_size=None, num_neg=None, batch_size=None):
        super(Word2Vec, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = window_size
        self.num_neg = num_neg
        self.batch_size = batch_size

        self.model = self._build_model()
        
    def get_last_layer_representation(self, word_index=None):
        # word_index: [batch_size, 1], batch of word_index
        # word_vec: normed representation of word

        # word_embedding: [batch_size, 1, embedding_dim]
        word_embedding = self.embedding_layer(word_index)

        # word_embedding: [batch_size, embedding_dim]
        word_embedding = tf.squeeze(word_embedding, axis=1)

        # word_dense: [batch_size, dense_units]
        word_dense = self.dense_layer(word_embedding)

        # === Normalize
        # word_norm: [batch_size, dense_units]
        word_norm = norm(word_dense, normed_axis=1)

        return word_norm

    def _build_model(self):
        inputs = Input(shape=(self.window_size*2, ))
        target = Input(shape=(1, ))
        negatives = Input(shape=(self.num_neg, ))

        emb_dim = 32
        dense_units = 32

        # Siamese Network
        self.embedding_layer = Embedding(input_dim=self.vocab_size,
                              output_dim=emb_dim)

        self.dense_layer = Dense(units=dense_units, activation='sigmoid')

        # === Embedding
        # inputs_embedding: [batch_size, input_len, output_dim]
        inputs_embedding = self.embedding_layer(inputs)

        # target_embedding: [batch_size, 1, output_dim]
        target_embedding = self.embedding_layer(target)

        # negatives_embedding: [batch_size, neg_len, output_dim]
        negatives_embedding = self.embedding_layer(negatives)

        # === Dense
        # inputs_dense: [batch_size, input_len, units]
        inputs_dense = self.dense_layer(inputs_embedding)

        # inputs_means: [batch_size, units]
        inputs_means = Lambda(lambda x: tf.reduce_mean(x, axis=1))(inputs_dense)

        # inputs_means: [batch_size, 1, units]
        inputs_means = tf.expand_dims(inputs_means, axis=1)

        # target_dense: [batch_size, 1, units]
        target_dense = self.dense_layer(target_embedding)

        # negatives_dense: [batch_size, neg_len, units]
        negatives_dense = self.dense_layer(negatives_embedding)

        # === Multiply
        # target_mul: [batch_size, 1, units]
        target_mul = tf.multiply(inputs_means, target_dense)

        # inputs_tile: [batch_size, neg_len, units]
        inputs_tile = tf.tile(inputs_means, multiples=[1, self.num_neg, 1])

        # negatives_mul: [batch_size, neg_len, units]
        negatives_mul = tf.multiply(inputs_tile, negatives_dense)

        # === Flatten
        # target_flatten: [batch_size, units]
        target_flatten = Flatten()(target_mul)

        # negatives_flatten: [batch_size, neg_len * units]
        negatives_flatten = Flatten()(negatives_mul)

        # concat_norm: [batch_size, (1 + neg_len) * units]
        concat_norm = tf.concat([target_flatten, negatives_flatten], axis=1)

        # === Softmax
        softmax = Dense(units=self.num_neg + 1, activation='softmax')(concat_norm)

        # === Model
        model = Model(inputs=[inputs, target, negatives], outputs=softmax)
        model.compile(optimizer=tf.optimizers.Adam(0.001),
                      loss=tf.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.metrics.sparse_categorical_accuracy])

        print(model.summary())
        return model

    def train(self,
              train_dataset=None,
              val_dataset=None,
              model_path=None,
              epochs=None,
              total_num_train=None,
              total_num_val=None,
              batch_size=None):

        batch_size = batch_size
        epochs = epochs
        total_num_train = total_num_train
        total_num_val = total_num_val

        steps_per_epoch = total_num_train // batch_size
        validation_steps = total_num_val // batch_size

        # === callbacks
        callbacks = []
        early_stopping_cb = EarlyStopping(monitor='val_loss', patience=5)
        callbacks.append(early_stopping_cb)

        model_checkpoint_cb = ModelCheckpoint(filepath=model_path,
                                              monitor='val_loss',
                                              save_best_only=True,
                                              save_weights_only=True)
        callbacks.append(model_checkpoint_cb)

        log_dir = os.path.join(get_logs_dir(), "word2vec", "tf_w2v_"
                               + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_cb)

        # === Model fit
        history = self.model.fit(train_dataset,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=val_dataset,
                       steps_per_epoch=steps_per_epoch,
                       validation_steps=validation_steps,
                       validation_batch_size=batch_size,
                       callbacks=callbacks)

        return history


    def call(self, inputs, target, negatives, training=None, mask=None):
        pass



if __name__ == "__main__":

    batch_size = 2
    vocab_size = 10

    input_len = 5
    negative_len = 4

    inputs = tf.random.uniform((batch_size, input_len), minval=0, maxval=vocab_size, dtype=tf.int32)
    target = tf.random.uniform((batch_size, 1), minval=0, maxval=vocab_size, dtype=tf.int32)
    negatives = tf.random.uniform((batch_size, negative_len), minval=0, maxval=vocab_size, dtype=tf.int32)

    w2v = Word2Vec(vocab_size=vocab_size)
    w2v(inputs, target, negatives)



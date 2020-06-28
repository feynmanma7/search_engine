import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from zhugeliang.utils.metrics import cos_sim, seq_cos_sim


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size=None):
        super(Word2Vec, self).__init__()

        self.vocab_size = vocab_size
        self.model = self._build_model()

    def _build_model(self):
        # === Embedding
        embedding = Embedding(input_dim=self.vocab_size,
                              output_dim=8)

        # inputs_embedding: [batch_size, input_len, output_dim]
        inputs_embedding = embedding(inputs)

        # target_embedding: [batch_size, 1, output_dim]
        target_embedding = embedding(target)

        # negatives_embedding: [batch_size, neg_len, output_dim]
        negatives_embedding = embedding(negatives)

        # === Dense
        # inputs_embedding: [batch_size, seq_len, units]
        # inputs_1: [batch_size, units]
        # Reduce mean for inputs
        inputs_1 = Lambda(lambda x: tf.reduce_mean(x, axis=1))(inputs_embedding)

        # inputs_dense: [batch_size, units]
        inputs_dense = Dense(units=4, activation='relu')(inputs_1)

        # target_dense: [batch_size, 1, units]
        target_dense = Dense(units=4, activation='relu')(target_embedding)
        # target_dense: [batch_size, units]
        target_dense = tf.squeeze(target_dense, axis=1)

        # [batch_size, neg_len, units]
        negatives_dense = Dense(units=4, activation='relu')(negatives_embedding)

        # === Cosine similarity
        # target_cos: [batch_size, 1]
        target_cos = cos_sim(inputs_dense, target_dense)

        # negatives_cos: [batch_size, neg_len]
        negatives_cos = seq_cos_sim(inputs_dense, negatives_dense)

        # concat_cos: [batch_size, 1 + neg_len]
        concat_cos = tf.concat([target_cos, negatives_cos], axis=1)

        # === Softmax
        # softmax: [batch_size, 1 + neg_len]
        softmax = tf.nn.softmax(concat_cos, axis=1)

        # === Model
        model = Model(inputs=[inputs, target, negatives], outputs=softmax)
        model.compile(optimizer=tf.optimizers.Adam(0.001),
                      loss=tf.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.metrics.categorical_crossentropy])
        return model

    def train(self, train_dataset=None, val_dataset=None, model_path=None):
        batch_size = 8
        epochs = 10
        total_num_train = 1000
        total_num_val = 200
        steps_per_epoch = total_num_train // batch_size
        validation_steps = total_num_val // batch_size

        # === callbacks
        callbacks = []
        early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10)
        callbacks.append(early_stopping_cb)

        model_checkpoint_cb = ModelCheckpoint(filepath=model_path,
                                              monitor='val_loss',
                                              save_best_only=True,
                                              save_weights_only=True)
        callbacks.append(model_checkpoint_cb)

        # === Model fit
        self.model.fit(train_dataset,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=val_dataset,
                       steps_per_epoch=steps_per_epoch,
                       validation_steps=validation_steps,
                       validation_batch_size=batch_size,
                       callbacks=callbacks)

        self.model.save_weights(model_path)


    def call(self, inputs, target, negatives, training=None, mask=None):
        # === Embedding
        embedding = Embedding(input_dim=self.vocab_size,
                              output_dim=8)

        # inputs_embedding: [batch_size, input_len, output_dim]
        inputs_embedding = embedding(inputs)

        # target_embedding: [batch_size, 1, output_dim]
        target_embedding = embedding(target)

        # negatives_embedding: [batch_size, neg_len, output_dim]
        negatives_embedding = embedding(negatives)

        # === Dense
        # inputs_embedding: [batch_size, seq_len, units]
        # inputs_1: [batch_size, units]
        # Reduce mean for inputs
        inputs_1 = Lambda(lambda x: tf.reduce_mean(x, axis=1))(inputs_embedding)

        # inputs_dense: [batch_size, units]
        inputs_dense = Dense(units=4, activation='relu')(inputs_1)

        # target_dense: [batch_size, 1, units]
        target_dense = Dense(units=4, activation='relu')(target_embedding)
        # target_dense: [batch_size, units]
        target_dense = tf.squeeze(target_dense, axis=1)

        # [batch_size, neg_len, units]
        negatives_dense = Dense(units=4, activation='relu')(negatives_embedding)

        # === Cosine similarity
        # target_cos: [batch_size, 1]
        target_cos = cos_sim(inputs_dense, target_dense)

        # negatives_cos: [batch_size, neg_len]
        negatives_cos = seq_cos_sim(inputs_dense, negatives_dense)

        # concat_cos: [batch_size, 1 + neg_len]
        concat_cos = tf.concat([target_cos, negatives_cos], axis=1)

        # === Softmax
        # softmax: [batch_size, 1 + neg_len]
        softmax = tf.nn.softmax(concat_cos, axis=1)

        return softmax



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



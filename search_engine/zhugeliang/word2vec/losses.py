import tensorflow as tf


def dssm_loss(y_true, y_pred):
    return tf.reduce_mean(-tf.math.log(y_pred[0]), axis=-1)
    #return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
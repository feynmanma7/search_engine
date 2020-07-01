import tensorflow as tf


def dssm_loss(y_true, y_pred):
    return -tf.math.log(y_pred[0])
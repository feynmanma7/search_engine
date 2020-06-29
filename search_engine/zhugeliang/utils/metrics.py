import tensorflow as tf
import numpy as np


def norm(a, normed_axis=1):
    # a: [batch_size, (seq_len), dim]
    # norm_a: [batch_size, (seq_len), ]
    norm_a = tf.norm(a, axis=normed_axis)

    # expand_norm: [batch_size, (seq_len), 1]
    expand_norm = tf.expand_dims(norm_a, axis=normed_axis)

    # broadcast_norm: [batch_size, (seq_len), dim]
    broadcast_norm = tf.broadcast_to(expand_norm, a.shape)

    # === divide by norm
    # normed_a: [batch_size, (seq_len), dim], same with a
    normed_a = tf.truediv(a, broadcast_norm)

    return normed_a


def cos_sim(a, b):
    # a: [batch_size, dim]
    # normed_a: [batch_size, dim]
    # ret: [batch_size, ]

    # a, b, TensorShape
    assert a.shape.as_list() == b.shape.as_list()

    normed_a = norm(a)

    # b: [batch_size, dim]
    # normed_b: [batch_size, dim]
    normed_b = norm(b)

    # ret: [batch_size, 1]
    return tf.expand_dims(tf.reduce_sum(tf.multiply(normed_a, normed_b), axis=1), axis=1)


def seq_cos_sim(a, b):
    # a: [batch_size, dim]
    # b: [batch_size, seq_len, dim]
    # ret: [batch_size, seq_len]

    # normed_a: [batch_size, dim]
    normed_a = norm(a, normed_axis=1)

    # normed_b: [batch_size, seq_len, dim]
    normed_b = norm(b, normed_axis=2)

    # normed_a: [batch_size, 1, dim]
    expand_a = tf.expand_dims(normed_a, axis=1)

    # broadcast_a: [batch_size, seq_len, dim]
    broadcast_a = tf.broadcast_to(expand_a, b.shape)

    # cos: [batch_size, seq_len]
    cos = tf.reduce_sum(tf.multiply(broadcast_a, normed_b), axis=2)

    return cos


if __name__ == '__main__':
    a = tf.constant([[3, 4]], dtype=tf.float32) # [1, 2]
    b = tf.constant([[6, 8]], dtype=tf.float32) # [1, 2]
    c = tf.constant([[[3, 4], [6, 8], [9, 18]]], dtype=tf.float32) # [1, 3, 2]

    print(a.shape, b.shape, c.shape)

    print(tf.multiply(a, b))

    # [1, 1, 2]
    aa = tf.expand_dims(a, axis=1)

    # [1, 3, 2]
    aa = tf.tile(aa, multiples=[1, 3, 1])
    print(aa.shape)
    print(aa)

    d = tf.multiply(aa, c)
    print(d)


    print(tf.reduce_mean(aa, axis=1).shape)
    """
    d = tf.constant([[3, 4], [6, 8]], dtype=tf.float32)
    print('norm', norm(d, normed_axis=1))
    print('tf_norm', tf.norm(a, axis=[0, 1], keepdims=True))

    print()
    print('cos_sim', cos_sim(a, b))
    print('tf_cos_sim', tf.keras.layers.Dot(axes=(1, 1), normalize=True)([a, b]))
    print('tf_cos_sim', tf.keras.layers.Dot(axes=(1, 1), normalize=True)([a, b]))
    print('tf_cos_sim, no norm', tf.keras.layers.Dot(axes=(1, 1), normalize=False)([a, b]))
    print()

    print('seq_cos_sim', seq_cos_sim(a, c))
    print('tf_cos_sim', tf.keras.layers.Dot(axes=(1, 2), normalize=True)([a, c]))
    print('tf_cos_sim, no norm', tf.keras.layers.Dot(axes=(1, 2), normalize=False)([a, c]))
    """


import unittest
import tensorflow as tf


class MyTest(tf.test.TestCase):
    def test_norm(self):
        from zhugeliang.utils.metrics import norm
        a = tf.constant([[3, 4]], dtype=tf.float32)
        b = norm(a)

        true_normed_a = tf.constant([[0.6, 0.8]], dtype=tf.float32)
        self.assertAllEqual(b, true_normed_a)

    def test_cos_sim(self):
        from zhugeliang.utils.metrics import cos_sim
        a = tf.constant([[3, 4]], dtype=tf.float32)
        b = tf.constant([[6, 8]], dtype=tf.float32)

        cos = cos_sim(a, b)
        true_cos = tf.constant([[1.0]], dtype=tf.float32) # [batch_size=1, 1]

        self.assertEqual(cos, true_cos)

    def test_seq_cos_sim(self):
        from zhugeliang.utils.metrics import seq_cos_sim
        a = tf.constant([[3, 4]], dtype=tf.float32) # [batch_size=1, dim=2]
        b = tf.constant([[[3, 4], [6, 8]]], dtype=tf.float32) # [batch_size=1, seq_len=2, dim=2]

        cos = seq_cos_sim(a, b)
        true_cos = tf.constant([[1.0, 1.0]], dtype=tf.float32) # [batch_size=1, seq_len=2]

        self.assertAllEqual(cos, true_cos)


if __name__ == "__main__":
    unittest.main()
from zhugeliang.word2vec.word2vec import Word2vec
from zhugeliang.utils.config import get_model_dir, get_data_dir
import os
from zhugeliang.word2vec.dataset import get_dataset
import time
import tensorflow as tf
import numpy as np


def train_word2vec():
    # vocab_size = 10001  # min_cnt=5, ptb
    # total_num_train = 971657
    # total_num_val = 77130

    vocab_size = 17617 # min_cnt=10, local_pdf
    total_num_train = 1615567
    total_num_val = 405872

    shuffle_buffer_size = 2048 * 2
    epochs = 10
    batch_size = 128
    window_size = 5
    num_neg = 5
    embedding_dim = 64 # To tune

    #train_path = os.path.join(get_data_dir(), "ptb.train.txt")
    #val_path = os.path.join(get_data_dir(), "ptb.valid.txt")
    #dict_dir = os.path.join(get_data_dir(), "book_dict")

    train_path = os.path.join(get_data_dir(), "shuf_train.txt")
    val_path = os.path.join(get_data_dir(), "shuf_val.txt")
    dict_dir = os.path.join(get_data_dir(), "book_dict")

    train_dataset = get_dataset(input_path=train_path,
                                dict_dir=dict_dir,
                                shuffle_buffer_size=shuffle_buffer_size,
                                epochs=epochs,
                                batch_size=batch_size,
                                window_size=window_size,
                                num_neg=num_neg)

    val_dataset = get_dataset(input_path=val_path,
                              dict_dir=dict_dir,
                              shuffle_buffer_size=shuffle_buffer_size,
                              epochs=epochs,
                              batch_size=batch_size,
                              window_size=window_size,
                              num_neg=num_neg)

    # === model
    model = Word2vec(vocab_size=vocab_size,
                     window_size=window_size,
                     num_neg=num_neg,
                     embedding_dim=embedding_dim)

    # === optimizer
    optimizer = tf.keras.optimizers.Adam(0.001)

    # === checkpoint
    checkpoint_dir = os.path.join(get_model_dir(), "word2vec")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    total_train_batch = total_num_train // batch_size + 1

    # === Train
    start = time.time()
    for epoch in range(epochs):
        total_loss = 0
        batch_loss = 0

        epoch_start = time.time()
        i = 0
        for batch_idx, (contexts, target, negatives) in zip(range(total_train_batch), train_dataset):
            i += 1

            # === self-defined batch train
            cur_loss = train_step(model, optimizer, contexts, target, negatives)
            batch_loss += cur_loss

            if i % 100 == 0:
                batch_end = time.time()
                batch_last = batch_end - start
                print("Epoch: %d/%d, batch: %d/%d, batch_loss: %.4f, cur_loss: %.4f, lasts: %.2fs" %
                      (epoch+1, epochs, batch_idx+1, total_train_batch, batch_loss/(batch_idx+1), cur_loss, batch_last))

        assert i > 0
        batch_loss /= i

        total_loss += batch_loss
        epoch_end = time.time()
        epoch_last = epoch_end - epoch_start
        print("Epoch: %d/%d, loss: %.4f, lasts: %.2fs" % (epoch+1, epochs, total_loss/(epoch+1), epoch_last))

        # === model save
        checkpoint.save(file_prefix=checkpoint_prefix)


    end = time.time()
    last = end - start
    print("Lasts %.2fs" % last)

@tf.function
def train_step(model, optimizer, contexts, target, negatives):

    with tf.GradientTape() as tape:
        batch_loss = model(contexts, target, negatives)

    variables = model.trainable_variables
    gradients = tape.gradient(batch_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


if __name__ == "__main__":
    train_word2vec()


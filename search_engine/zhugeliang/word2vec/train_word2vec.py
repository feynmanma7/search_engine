from zhugeliang.word2vec.word2vec import Word2vec
from zhugeliang.utils.config import get_model_dir, get_data_dir
import os
from zhugeliang.word2vec.dataset import get_dataset
import time
import tensorflow as tf


def train_word2vec():
    vocab_size = 10001  # min_cnt=5, ptb
    total_num_train = 971657
    total_num_val = 77130

    shuffle_buffer_size = 2048 * 2
    epochs = 100
    batch_size = 128
    window_size = 3
    num_neg = 5
    embedding_dim = 8 # To tune

    train_path = os.path.join(get_data_dir(), "ptb.train.txt")
    val_path = os.path.join(get_data_dir(), "ptb.valid.txt")
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

    optimizer = tf.keras.optimizers.Adam(0.001)

    model = Word2vec(vocab_size=vocab_size,
                     window_size=window_size,
                     num_neg=num_neg,
                     embedding_dim=embedding_dim)


    # === Train
    for epoch in range(epochs):
        total_loss = 0
        batch_loss = 0

        i = 0
        for batch_idx, (contexts, target, negatives) in enumerate(train_dataset):
            i += 1
            batch_loss += train_step(model, optimizer, contexts, target, negatives)
            print("Epoch: %d/%d, batch: %d, loss: %.4f" %
                  (epoch+1, epochs, batch_idx, batch_loss/(batch_idx+1)))

        assert i > 0
        batch_loss /= i

        total_loss += batch_loss
        print(epoch, total_loss/(epoch+1))


@tf.function
def train_step(model, optimizer, contexts, target, negatives):

    with tf.GradientTape() as tape:
        batch_loss = model(contexts, target, negatives)

    #batch_loss = batch_loss / contexts.shape[0] # need?

    variables = model.trainable_variables
    gradients = tape.gradient(batch_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

if __name__ == "__main__":
    train_word2vec()

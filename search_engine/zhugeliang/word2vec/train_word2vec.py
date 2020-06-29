from zhugeliang.word2vec.word2vec import Word2Vec
from zhugeliang.utils.config import get_model_dir
import os
from zhugeliang.word2vec.dataset import get_train_dataset, get_val_dataset
import time

if __name__ == "__main__":
    shuffle_buffer_size = 1024
    epochs = 10
    batch_size = 32
    vocab_size = 14830
    total_num_train = 86308
    total_num_val = 21694

    train_dataset = get_train_dataset(epochs=epochs,
                                      shuffle_buffer_size=shuffle_buffer_size,
                                      batch_size=batch_size)
    val_dataset = get_val_dataset(epochs=epochs,
                                  shuffle_buffer_size=shuffle_buffer_size,
                                  batch_size=batch_size)

    model_path = os.path.join(get_model_dir(), "word2vec", "ckpt")

    start = time.time()

    window_size = 3
    num_neg = 4

    w2v = Word2Vec(vocab_size=vocab_size,
                   window_size=window_size,
                   num_neg=num_neg,
                   batch_size=batch_size)

    w2v.train(train_dataset=train_dataset,
              val_dataset=val_dataset,
              model_path=model_path,
              epochs=epochs,
              total_num_train=total_num_train,
              total_num_val=total_num_val,
              batch_size=batch_size)

    end = time.time()
    last = end - start
    print("Word2vec train done! Lasts %.2fs" % last)
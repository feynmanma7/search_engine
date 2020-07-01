import numpy as np
from zhugeliang.word2vec.dictionary import load_dictionary
from zhugeliang.utils.config import get_data_dir
import os


class Sampler:
    def __init__(self, word_cnt_dict=None, id2word_dict=None):
        #self.num_word = word_cnt_dict
        self.word_idxes = list(range(len(id2word_dict)))

        counts = np.array([word_cnt_dict[id2word_dict[idx]] for idx in self.word_idxes])
        probs = counts / np.sum(counts)
        self.probs = np.concatenate(([0], np.cumsum(probs)))


    def sample(self, num_sample=None, method="random"):
        if method == "random":
            return self._random_sample(num_sample)

    def _random_sample(self, num_sample):
        return np.random.choice(self.word_idxes, num_sample)

    def _weighted_sample(self, num_sample):
        r = np.random.random(num_sample)
        choices = []
        for j in range(10):
            pass


if __name__ == '__main__':
    dict_dir = os.path.join(get_data_dir(), "book_dict")

    word_cnt_dict_path = os.path.join(dict_dir, "word_cnt_dict.pkl")
    word_cnt_dict = load_dictionary(dict_path=word_cnt_dict_path)
    print(len(word_cnt_dict))

    word2id_dict_path = os.path.join(dict_dir, "word2id_dict.pkl")
    word2id_dict = load_dictionary(dict_path=word2id_dict_path)
    print(len(word2id_dict))

    id2word_dict_path = os.path.join(dict_dir, "id2word_dict.pkl")
    id2word_dict = load_dictionary(dict_path=id2word_dict_path)
    print(len(id2word_dict))

    #word_cnt_dict = {"word": 100, 2: 200, 3: 300}
    #word2id_dict = {"machine": 0, 2: 1, 3: 2}
    sampler = Sampler(word_cnt_dict=word_cnt_dict, id2word_dict=id2word_dict)

    samples = sampler.sample(num_sample=10, method="random")
    print(samples)

    samples = sampler.sample(num_sample=10, method="weighted")
    print(samples.probs)

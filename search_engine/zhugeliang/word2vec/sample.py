import numpy as np


class Sampler:
    def __init__(self, word_cnt_dict=None, word2id_dict=None):
        #self.num_word = word_cnt_dict
        self.words = list(range(len(word2id_dict)))

    def sample(self, num_sample=None, method="random"):
        if method == "random":
            return self._random_sample(num_sample)

    def _random_sample(self, num_sample):
        return np.random.choice(self.words, num_sample)


if __name__ == '__main__':
    word_cnt_dict = {1: 100, 2: 200, 3: 300}
    word2id_dict = {1: 0, 2: 1, 3: 2}
    sampler = Sampler(word_cnt_dict=word_cnt_dict, word2id_dict=word2id_dict)

    samples = sampler.sample(num_sample=10, method="random")
    print(samples)
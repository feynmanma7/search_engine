from zhugeliang.utils.config import get_data_dir
from zhugeliang.word2vec.dictionary import load_dictionary
import os


def count_training_num(input_path=None, word_cnt_dict=None, min_word_cnt=None):
    # input_path: word '\s' word '\s'

    with open(input_path, 'r', encoding='utf-8') as f:

        training_num = 0

        for line in f:
            for word in line[:-1].split(' '):
                if word not in word_cnt_dict:
                    continue
                cnt = word_cnt_dict[word]

                if cnt < min_word_cnt:
                    continue

                training_num += 1

        return training_num


if __name__ == '__main__':
    train_path = os.path.join(get_data_dir(), "shuf_val.txt")
    #train_path = os.path.join(get_data_dir(), "ptb.valid.txt")
    word_cnt_dict_path = os.path.join(get_data_dir(), "book_dict", "word_cnt_dict.pkl")
    min_word_cnt = 10

    word_cnt_dict = load_dictionary(dict_path=word_cnt_dict_path)
    print(len(word_cnt_dict))

    training_num = count_training_num(input_path=train_path,
                                      word_cnt_dict=word_cnt_dict,
                                      min_word_cnt=min_word_cnt)
    print(training_num)


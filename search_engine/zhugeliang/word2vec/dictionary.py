from zhugeliang.utils.config import get_data_dir
import os, pickle


def load_stop_words_dict(stop_words_dict_path=None):

    with open(stop_words_dict_path, 'r', encoding='utf-8') as f:
        stop_words_dict = {}
        for line in f:
            stop_word = line[:-1]
            stop_words_dict[stop_word] = True

        return stop_words_dict

    return {}


def load_dictionary(dict_path=None):
    _dict = {}
    with open(dict_path, 'rb') as fr:
        _dict = pickle.load(fr)

        return _dict

    return {}


def build_dictionary(dict_dir=None, min_word_count=None):
    word_cnt_dict_path = os.path.join(dict_dir, "word_cnt_dict.pkl")
    with open(word_cnt_dict_path, 'rb') as fr:
        word_cnt_dict = pickle.load(fr)

    assert word_cnt_dict is not None

    word2id_dict, id2word_dict = {}, {}

    # === Remove the word of word_count < min_word_count.
    word_index = 0
    for word, cnt in sorted(word_cnt_dict.items(), key=lambda x: -x[1]):
        if cnt < min_word_count:
            break

        if word not in word2id_dict:
            word2id_dict[word] = word_index
            id2word_dict[word_index] = word
            word_index += 1

    # === Store word2id and id2word
    word2id_dict_path = os.path.join(dict_dir, "word2id_dict.pkl")
    with open(word2id_dict_path, 'wb') as fr:
        pickle.dump(word2id_dict, fr)

    id2word_dict_path = os.path.join(dict_dir, "id2word_dict.pkl")
    with open(id2word_dict_path, 'wb') as fr:
        pickle.dump(id2word_dict, fr)

    return True


def count_word(text_path=None, dict_path=None):
    """
    Count and store the word_count dict.
    """

    # === Compute word_count
    word_cnt_dict = {}
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line[:-1]) == 0:
                continue

            for word in line.split(' '):
                if word not in word_cnt_dict:
                    word_cnt_dict[word] = 1
                else:
                    word_cnt_dict[word] += 1

    # === Save word_cnt_dict
    with open(dict_path, 'wb') as fw:
        pickle.dump(word_cnt_dict, fw)


if __name__ == "__main__":
    #text_path = os.path.join(get_data_dir(), "book_text.txt")
    text_path = os.path.join(get_data_dir(), "ptb_train_val.txt")
    dict_dir = os.path.join(get_data_dir(), "book_dict")

    word_cnt_dict_path = os.path.join(dict_dir, "word_cnt_dict.pkl")

    count_word(text_path=text_path, dict_path=word_cnt_dict_path)

    build_dictionary(dict_dir=dict_dir, min_word_count=5)

    word_cnt_dict = load_dictionary(dict_path=word_cnt_dict_path)
    print(len(word_cnt_dict))

    word2id_dict_path = os.path.join(dict_dir, "word2id_dict.pkl")
    word2id_dict = load_dictionary(dict_path=word2id_dict_path)
    print(len(word2id_dict))
    id2word_dict_path = os.path.join(dict_dir, "id2word_dict.pkl")
    id2word_dict = load_dictionary(dict_path=id2word_dict_path)
    print(len(id2word_dict))

    i = 0
    for i, (word, idx) in zip(range(10), word2id_dict.items()):
        print(word, "\t", idx, "\t", word_cnt_dict[word])
    print('-' * 30, '\n')
    for _, (word, cnt) in zip(range(10), sorted(word_cnt_dict.items(), key=lambda x: -x[1])):
        print(word, cnt)

    #word_cnt_dict, word2id_dict, id2word_dict = load_dictionary(dict_dir)
    #print(len(word_cnt_dict), len(word2id_dict), len(id2word_dict))
from zhugeliang.utils.config import get_data_dir
from zhugeliang.utils.text_process import process_text
import os, pickle


def load_dictionary(dict_dir=None):
    word_cnt_dict_path = os.path.join(dict_dir, "word_cnt_dict.pkl")
    with open(word_cnt_dict_path, 'rb') as fr:
        word_cnt_dict = pickle.load(fr)

    word2id_dict_path = os.path.join(dict_dir, "word2id_dict.pkl")
    with open(word2id_dict_path, 'rb') as fr:
        word2id_dict = pickle.load(fr)

    id2word_dict_path = os.path.join(dict_dir, "id2word_dict.pkl")
    with open(id2word_dict_path, 'rb') as fr:
        id2word_dict = pickle.load(fr)

    return word_cnt_dict, word2id_dict, id2word_dict


def build_dictionary(dict_dir=None, min_word_count=None):

    word_cnt_dict_path = os.path.join(dict_dir, "word_cnt_dict.pkl")
    with open(word_cnt_dict_path, 'rb') as fr:
        word_cnt_dict = pickle.load(fr)

    assert word_cnt_dict is not None

    word2id_dict, id2word_dict = {}, {}

    # === Remove the word of word_count < min_word_count.
    word_index = 0
    for word, cnt in word_cnt_dict.items():
        if cnt < min_word_count:
            continue

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


def count_word(raw_text_path, seg_text_path, dict_dir):
    """
    Segmentation on raw text, write segmentation-ed text into disk,
    and store the word_count dict.
    """

    # Generate and save word_count dict
    fw = open(seg_text_path, 'w', encoding='utf-8')

    word_cnt_dict = {}

    with open(raw_text_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line[:-1]) == 0:
                continue

            words = []
            for word in process_text(line[:-1]):
                if word in [" ", "\t", "\n"]:
                    continue

                words.append(word)
                if word not in word_cnt_dict:
                    word_cnt_dict[word] = 1
                else:
                    word_cnt_dict[word] += 1

            fw.write(" ".join(words) + "\n")

    fw.close()

    dict_path = os.path.join(dict_dir, "word_cnt_dict.pkl")
    with open(dict_path, 'wb') as fw:
        pickle.dump(word_cnt_dict, fw)


if __name__ == "__main__":
    raw_text_path = os.path.join(get_data_dir(), "book_text.txt")
    seg_text_path = os.path.join(get_data_dir(), "book_seg_text.txt")
    dict_dir = os.path.join(get_data_dir(), "book_dict")

    #count_word(raw_text_path, seg_text_path, dict_dir)
    #print("Write word_count_dict done! %s" % seg_text_path)

    #build_dictionary(dict_dir=dict_dir, min_word_count=5)
    #print("Write word2id_dict and id2word_dict done!")

    word_cnt_dict, word2id_dict, id2word_dict = load_dictionary(dict_dir)
    print(len(word_cnt_dict), len(word2id_dict), len(id2word_dict))
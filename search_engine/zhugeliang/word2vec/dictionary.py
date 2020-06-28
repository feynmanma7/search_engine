from zhugeliang.utils.config import get_data_dir
from zhugeliang.utils.text_process import process_text
import os, pickle


def load_dictionary(dict_path):
    with open(dict_path, 'rb') as fr:
        word_cnt_dict = pickle.load(fr)

        return word_cnt_dict

    return None


def build_dictionary(raw_text_path, seg_text_path, dict_path):
    """
    Segmentation on raw text, write segmentation-ed text into disk,
    and store the word_count dict.
    """
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

    with open(dict_path, 'wb') as fw:
        pickle.dump(word_cnt_dict, fw)


if __name__ == "__main__":
    raw_text_path = os.path.join(get_data_dir(), "book_text.txt")
    seg_text_path = os.path.join(get_data_dir(), "book_seg_text.txt")
    dict_path = os.path.join(get_data_dir(), "book_dict.pkl")

    build_dictionary(raw_text_path, seg_text_path, dict_path)
    print("Write done! %s" % seg_text_path)
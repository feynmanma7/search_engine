from zhugeliang.utils.pdf_extractor import pdf_extractor
from zhugeliang.utils.text_process import process_text
from zhugeliang.utils.config import get_book_data_dir, get_data_dir
from zhugeliang.word2vec.dictionary import load_stop_words_dict
import glob, os
import traceback
import time


def extract_text(data_path=None, text_path=None, stop_words_dict=None):
    fw = open(text_path, 'w', encoding='utf-8')

    total_file_cnt = 0
    file_cnt = 0
    wrong_cnt = 0
    for file_path in glob.glob(data_path, recursive=True):
        total_file_cnt += 1

        try:
            is_write = False

            for text in pdf_extractor(file_path=file_path, max_page_number=10):

                if text is None or len(text) == 1:
                    continue

                # One page of text a line.
                words = []
                for word in process_text(text):

                    # === One-Character. Remove all of the length-ONE word, ' ', '\t', '\n' included.
                    if len(word) == 1:
                        continue

                    # === Digit. Remove digit-starting or -[digit]-starting word.
                    if word[0].isdigit() or (word[0] == '-' and word[1].isdigit()):
                        continue

                    # === Stop words
                    if word in stop_words_dict:
                        continue

                    words.append(word)

                if len(words) < 7: # window_size * 2 + 1 = 3 * 2 + 1
                    continue

                fw.write(' '.join(words) + '\n')
                is_write = True

            if is_write:
                file_cnt += 1
                print(file_cnt, "\t", file_path)
            else:
                print("Empty\t", file_path)

        except:
            wrong_cnt += 1
            print("Wrong No. %d\t%s" % (wrong_cnt, file_path))
            traceback.print_exc()
            print('\n' * 3)

    print("\n\nWrite Done.")
    print("Total #file = %d" % total_file_cnt)
    print("Total #write_file = %d" % file_cnt)
    print("Total #wrong = %d" % wrong_cnt)


if __name__ == "__main__":
    book_data_path = os.path.join(get_book_data_dir(), "**/*.pdf")
    text_path = os.path.join(get_data_dir(), "book_text.txt")

    # === Load stop words dict.
    stop_words_dict_path = os.path.join(get_data_dir(), "baidu_stopwords.txt")
    stop_words_dict = load_stop_words_dict(stop_words_dict_path=stop_words_dict_path)
    print("#stop_words = %d" % len(stop_words_dict))

    # === Extract text.
    start = time.time()
    extract_text(data_path=book_data_path, text_path=text_path, stop_words_dict=stop_words_dict)
    end = time.time()
    last = end - start
    print("Lasts %.2fs" % last)


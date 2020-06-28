from zhugeliang.search_engine.seach_engine import SearchEngine
from zhugeliang.utils.config import get_book_data_dir, get_model_dir
import os

if __name__ == '__main__':
    dir_path = get_book_data_dir()
    model_dir_path = os.path.join(get_model_dir(), "v1")

    se = SearchEngine(max_page_number=10)
    dir_path = os.path.join(dir_path, "**/*.pdf")
    se.build_inverted_index(dir_path=dir_path,
                            model_dir_path=model_dir_path)

    print(se.inverted_index)

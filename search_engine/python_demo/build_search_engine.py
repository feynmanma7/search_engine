from python_demo.seach_engine import SearchEngine
import sys

if __name__ == '__main__':
    """
    if len(sys.argv) < 2:
        print('Less arguments!')
        sys.exit(0)
    """

    #dir_path = sys.argv[1]
    #index_path = sys.argv[2]

    dir_path = "/Users/flyingman/Book/machine learning"
    model_dir_path = "/Users/flyingman/Developer/github/search_engine/model"

    se = SearchEngine()

    se.build_inverted_index(dir_path=dir_path,
                            model_dir_path=model_dir_path)

    print(se.inverted_index)

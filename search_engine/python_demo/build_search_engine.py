from python_demo.seach_engine import SearchEngine
import sys, os

if __name__ == '__main__':
    """
    if len(sys.argv) < 2:
        print('Less arguments!')
        sys.exit(0)
    """

    #dir_path = sys.argv[1]
    #index_path = sys.argv[2]

    #dir_path = "/Users/flyingman/Book/machine learning"
    dir_path = "/Users/flyingman/Book/"
    model_dir_path = "/Users/flyingman/Developer/github/search_engine/model/v1"

    se = SearchEngine(max_page_number=10)
    dir_path = os.path.join(dir_path, "**/*.pdf")
    se.build_inverted_index(dir_path=dir_path,
                            model_dir_path=model_dir_path)

    print(se.inverted_index)

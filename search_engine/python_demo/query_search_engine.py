from python_demo.seach_engine import SearchEngine

if __name__ == '__main__':
    dir_path = "/Users/flyingman/Book/machine learning"
    model_dir_path = "/Users/flyingman/Developer/github/search_engine/model"

    se = SearchEngine()
    se.load_model(model_dir_path=model_dir_path)
    print("Load model done!")

    query = "Machine Learning"
    ans = se.search(query=query)
    print(ans)


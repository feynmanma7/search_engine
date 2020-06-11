from python_demo.seach_engine import SearchEngine

if __name__ == '__main__':
    dir_path = "/Users/flyingman/Book/machine learning"
    model_dir_path = "/Users/flyingman/Developer/github/search_engine/model"

    se = SearchEngine()
    se.load_model(model_dir_path=model_dir_path)
    print("Load model done!")

    id2doc_dict = {doc_id: doc_path for doc_path, doc_id in se.doc2id_dict.items()}

    query = "computer vision"
    doc_id_list = se.search(query=query)

    doc_path_list = [id2doc_dict[doc_id] for doc_id in doc_id_list]
    for doc_path in doc_path_list:
        print(doc_path)


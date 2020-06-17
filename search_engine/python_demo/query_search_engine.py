from python_demo.seach_engine import SearchEngine
import sys

if __name__ == '__main__':
    #dir_path = "/Users/flyingman/Book/machine learning"
    model_dir_path = "/Users/flyingman/Developer/github/search_engine/model/v1"

    se = SearchEngine()
    se.load_model(model_dir_path=model_dir_path)
    print("Load model done!")

    id2doc_dict = {doc_id: doc_path for doc_path, doc_id in se.doc2id_dict.items()}

    query = "机器学习"
    doc_id_list = se.search(query=query)

    if doc_id_list is None:
        sys.exit(0)

    print("Total number of files is: %d" % len(doc_id_list))


    doc_path_list = [id2doc_dict[doc_id] for doc_id in doc_id_list]
    for i, doc_path in enumerate(doc_path_list):
        print(i, '/'.join(doc_path.split('/')[4:]))


<h1>word2vec</h1>

# 1. extract_text

Extract and preprocess pdf files, get segmentated files.

# 2. dictionary

> count_word, get word_cnt dictionary.
> build_dictionary, build word2id_dict and id2word_dict according to min_word_count.

# 3. train_val_split

`zhugeliang.utils.train_val_split`

Split the segmentated file into train and validation files.

# 4. train_word2vec

Train word2vec model.

# 5. similar_words

Find top-N similar words.

import jieba


def process_word(word):
    word = word.lower()

    return word


def process_text(text=None, max_word_len=30):
    # Segmentation
    for word in jieba.cut(text):
        if word is None or len(word) == 0:
            continue

        word = process_word(word)

        if len(word) > max_word_len:
            continue

        yield word


def process_english_text(text=None, max_word_len=30):
    for word in text.split(' '):
        word = word.lower()
        if len(word) > max_word_len:
            continue

        yield word
def process_english_text(text=None, max_word_len=30):
    for word in text.split(' '):
        word = word.lower()
        if len(word) > max_word_len:
            continue

        yield word
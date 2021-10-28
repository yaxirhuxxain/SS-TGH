# -*- coding: utf-8 -*-

# Author Yasir Hussain (yaxirhuxxain@yahoo.com)


import re

def encoder(word_idx, item):
    return [word_idx[word] if word in word_idx else word_idx['<idf>'] for word in item.split()]


def vocab_builder(files, word_idx, remove_singleton=True, vocab_limit: int = None):
        print("\t > Reading Files")
        train_data = ""
        for file in files:
            train_data = train_data + open(file, "r", encoding="utf-8").read()
            train_data = train_data + "\n"
        tokens_stream = re.sub(r"\n", " ", train_data).split()  # split data into tokens to build vocab

        print("\t > Building Word Count")
        word_count = {}
        for word in tokens_stream:
            try:
                word_count[word] += 1
            except:
                word_count[word] = 1

        if remove_singleton:
            print("\t > Remove Singleton")
            word_count = {k: v for k, v in word_count.items() if v > 1}

        print("\t > Sort Word Count")
        word_count = dict(sorted(word_count.items(), key=lambda kv: kv[1], reverse=True))

        print("\t > Adding to Global Vocabulary")
        count = len(word_idx)
        for word in word_count.keys():
            if vocab_limit and count >= vocab_limit:
                print(f"\t > Limit Vocab Size: {count}")
                break

            if word in word_idx:
                continue

            word_idx[word] = count
            count += 1

        vocab_size = len(word_idx)
        return word_idx, vocab_size

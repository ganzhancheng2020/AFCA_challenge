import jieba


def key_word_score(str1, str2):
    count = 0
    key_words = set(jieba.lcut(str1))
    list_ = jieba.lcut(str2)
    for word in list_:
        if word in key_words:
            count += 1

    return count/len(list_)
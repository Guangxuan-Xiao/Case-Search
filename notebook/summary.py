from hashlib import new
import math
import nltk
from nltk import PorterStemmer
import re
import jieba
from collections import OrderedDict
FINGERPRINT_LEN = 20

with open('/home/xgx/search-engine-project/baseline/stopword.txt') as f:
    stopwords = [i.strip() for i in f.readlines()]
stopwords = set(stopwords + ['.', '（', '）', '-'])

with open('/home/xgx/search-engine-project/data/startwords.txt') as f:
    startwords = [i.strip() for i in f.readlines()]
startwords = set(startwords)


def get_text_after_last_startword(text):
    start_from = 0
    for startword in startwords:
        last_occurence = text.rfind(startword)
        if last_occurence != -1:
            new_start_from = last_occurence + len(startword)
            if new_start_from > start_from:
                start_from = new_start_from
    return text[start_from:]


def split_sentence(text):
    text = text.strip()
    for sent in re.findall(u'[^!?。：；、\!\?]+[!?。：、；\!\?]?', text, flags=re.U):
        sent = sent.strip()
        if sent != '':
            yield sent


def unique_sentences(sentences):
    sentence_dict = OrderedDict()
    for sent in sentences:
        fingerprint = sent[:FINGERPRINT_LEN]
        if fingerprint not in sentence_dict:
            sentence_dict[fingerprint] = sent
    return list(sentence_dict.values())


def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = jieba.cut(sent, cut_all=False)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopwords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:FINGERPRINT_LEN]] = freq_table

    return frequency_matrix


def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / (count_words_in_sentence + 1e-5)

        tf_matrix[sent] = tf_table

    return tf_matrix


def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(
                total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def _score_sentences(tf_idf_matrix) -> list:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: list
    """

    sentenceValue = []

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue.append(total_score_per_sentence /
                             (count_words_in_sentence + 1e-5))

    return sentenceValue


def score_sentences(sentences) -> list:
    freq_matrix = _create_frequency_matrix(sentences)
    tf_matrix = _create_tf_matrix(freq_matrix)
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    total_documents = len(freq_matrix)
    idf_matrix = _create_idf_matrix(
        freq_matrix, count_doc_per_words, total_documents)
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    sentence_scores = _score_sentences(tf_idf_matrix)

    return sentence_scores

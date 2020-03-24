"""
The module includes all functions needed for text processing, and performing feature extraction with the
hashing trick (hashing trick is used instead of TF-IDF because of the size of the training set

Note: some of the functions are not used in the model because of very long pre-processing time (mainly removing
stop-words and normalizing all the words)
"""

import pandas as pd
import re
import os
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import HashingVectorizer


def process_text(text):
    """
    Takes a text (string) and performs all the required pre-processing steps one by one.
    Important note: all last 3 functions were written but not implemented because of time complexity limitations.
    :param text: the text to perform all the manipulations on
    :return: the text after manipulation
    """
    text = switch_exclamation_questionmark_signs(text)
    text = remove_punctuation(text)
    # text = lowercase_capitalization_words(text)
    # text = remove_stop_words(text, download=False)
    # text = normalize_words(text, download=False)
    return text


def switch_exclamation_questionmark_signs(text):
    """
    Switches '!' and '?' signs with the equivalent words, so that they will be processed by the hashing vectorizer
    :param text: string
    :return: the string after manipulation
    """
    text = text.replace('!', " exclamation")
    text = text.replace('?', " questionmark")
    return text


def remove_punctuation(text):
    """
    Remove all punctuation from the text
    :param text: string
    :return: the text after removing punctuations
    """
    text = text.replace('<br />', ' ')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def lowercase_capitalization_words(text):
    """
    Differentiate between words written in capital letters to capitalized words. Lowers every word that only her first
    letter is a capital letter
    :param text: string
    :return: after changing the words with capital letter only to lowercase
    """
    new_text = []
    for word in text.split():
        if word[0].isupper():
            lowercase_word = (word[0].lower() + word[1:])
            if lowercase_word == word.lower():
                word = lowercase_word
        new_text.append(word)
    return ' '.join(new_text)


def remove_stop_words(text, download):
    """
    Removing stop-words with nltk library
    :param text: string
    :param download: if true, it downloads the stop-words library
    :return: after removing the stop-words
    """
    if download:
        nltk_download('stopwords')
    words_list_without_stop_words = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(words_list_without_stop_words)


def normalize_words(text, download):
    """
    Normalizing words with nltk library
    :param text: string
    :param download: if true, it downloads the normalization dictionary
    :return: after changing words to their "basic" form
    """
    if download:
        nltk_download('wordnet')
    lemmatizer = WordNetLemmatizer()
    lemmatized_words_list = [lemmatizer.lemmatize(word) for word in text.split()]
    return ' '.join(lemmatized_words_list)


def text_preprocess(file_path, features_list, output_file):
    """
    Implementing the text process procedure and copy the value to a new csv file
    :param file_path: file path to be processed
    :param features_list: the features in the file
    :param output_file: the output file path
    :return: outputs a csv file, returns void
    """
    for chunk in pd.read_csv(file_path, chunksize=10**5, names=features_list):
        chunk['review/text'] = chunk['review/text'].apply(process_text)
        chunk.to_csv(output_file, mode='a', header=False, index=False)
    os.remove(file_path)


def feature_extraction_from_text(text_reviews, n_features):
    """
    Extracting features from existing corpus with the hashing trick (returns a sparse matrix of values)
    :param text_reviews: list of strings - the corpus
    :param n_features: number of features for each row of the sparse matrix
    :return: the sparse matrix that represent the features of each string in the corpus
    """
    reviews_list_of_lists = text_reviews.astype(str).values.tolist()
    # turn the list of lists to one list with strings
    reviews_in_list = [review for sublist in reviews_list_of_lists for review in sublist]
    # performing the hashing trick on the corpus
    vectorizer = HashingVectorizer(n_features=n_features)
    text_features = vectorizer.fit_transform(reviews_in_list)
    return text_features

import pandas as pd
import re
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def process_text(text):
    text = switch_exclamation_questionmark_signs(text)
    text = remove_punctuation(text)
    # text = lowercase_capitalization_words(text)
    # text = remove_stop_words(text, download=False)
    # text = normalize_words(text, download=False)
    return text


def switch_exclamation_questionmark_signs(text):
    text = text.replace('!', " exclamation")
    text = text.replace('?', " questionmark")
    return text


def remove_punctuation(text):
    text = text.replace('<br />', ' ')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def lowercase_capitalization_words(text):
    new_text = []
    for word in text.split():
        if word[0].isupper():
            lowercase_word = (word[0].lower() + word[1:])
            if lowercase_word == word.lower():
                word = lowercase_word
        new_text.append(word)
    return ' '.join(new_text)


def remove_stop_words(text, download):
    if download:
        nltk_download('stopwords')
    words_list_without_stop_words = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(words_list_without_stop_words)


def normalize_words(text, download):
    if download:
        nltk_download('wordnet')
    lemmatizer = WordNetLemmatizer()
    lemmatized_words_list = [lemmatizer.lemmatize(word) for word in text.split()]
    return ' '.join(lemmatized_words_list)

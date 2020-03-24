"""
This module prepares the data for the text preprocessing and for the training process.
Includes two main functions: converting txt to csv, and splitting an existing csv to two files randomly.
"""

import pandas as pd
import numpy as np
import os


def convert_txt_to_csv(txt_path, csv_path, features_list):
    """
    Converts a txt to csv file by a list of requested features - only the requested features will be extracted
    :param txt_path: the path of the txt file
    :param csv_path: the path of the csv file
    :param features_list: list of features to be converted to the csv
    :return: outputs the csv file, returns void
    """
    number_of_features = len(features_list)
    features_and_values = {}
    with open(txt_path, 'r', encoding='ISO-8859-1') as file, open(csv_path, 'w') as output:
        for line in file:
            # remove commas from text
            line_no_commas = line.replace(',', ' ')
            # extract the features and their values to a dictionary
            features_and_values = extract_feature_from_line_to_dict(features_list, line_no_commas, features_and_values)
            # when the dictionary has all the requested features, write the line to the output file and begin a new one
            if len(features_and_values) == number_of_features:
                written_line = write_dictionary_values_to_string_by_order(features_and_values, features_list)
                output.write(written_line)
                features_and_values = {}


def split_dataset(file_path, first_split_path, second_split_path, percentage, chunksize, random):
    """
    Splitting two csv files by requested percentage, can be used to split train-validation-test sets.
    Splitting the files randomly, with an option to define a seed
    :param file_path: file to split
    :param first_split_path: first output file path
    :param second_split_path: second output file path
    :param percentage: the percentage of the first file from the splitting file
    :param chunksize: fixed chunksize for iterating the dataset
    :param random: the random seed
    :return: outputs the two new files, returns void
    """
    for chunk in pd.read_csv(file_path, chunksize=chunksize, header=None):
        # using a seed to produce the same data sets if required (based on same starting set)
        np.random.seed(random)
        random_column_for_split = np.random.randint(0, 100, size=len(chunk))  # last chunk is smaller so size=len
        # creating a column with random numbers, then splitting the rows by the values of the random column
        chunk['split'] = random_column_for_split
        first_chunk = chunk[chunk.split < (percentage * 100)]
        second_chunk = chunk[chunk.split >= (percentage * 100)]

        # save into two distinct files, without the random column
        first_chunk.drop(columns=['split']).to_csv(first_split_path, mode='a', header=False, index=False)
        second_chunk.drop(columns=['split']).to_csv(second_split_path, mode='a', header=False, index=False)
    # remove the first file after splitting him
    os.remove(file_path)


def extract_feature_from_line_to_dict(features_list, line, dictionary):
    """
    Extracting the requested features and their values to a dictionary.
    If it finds a requested feature in the line, it will add him to the dictionary.
    If not, will return the dictionary without change
    :param features_list: the requested features
    :param line: string
    :param dictionary: existing dictionary
    :return: the dictionary
    """
    find_colon = line.find(":")
    feature_name = line[:find_colon]
    # if no feature exists or the line is empty, return the same dictionary
    if find_colon == -1 or feature_name not in features_list:
        return dictionary
    # else, append a feature and its values
    feature_value = line[find_colon + 1:].strip()
    dictionary[feature_name] = feature_value
    return dictionary


def write_dictionary_values_to_string_by_order(dictionary, ordered_list):
    """
    Writing the dictionary values to a string by order
    :param dictionary: the dictionary
    :param ordered_list: the order to write the values
    :return: the string
    """
    line = ""
    for item in ordered_list:
        feature_value = dictionary[item] + ','
        line += feature_value
    # remove last ',' to be read by the csv file
    line = line[:-1]
    line += '\n'
    return line


def check_csv_without_nulls(csv_path):
    """
    Checks whether a csv has any Null (NaN in Pandas) values
    :param csv_path: the csv path
    :return: True if yes, false otherwise
    """
    for chunk in pd.read_csv(csv_path, chunksize=10**5):
        if chunk.isnull().values.any():
            return True
    return False

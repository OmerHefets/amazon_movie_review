def convert_txt_to_csv(txt_path, csv_path, features_list):
    number_of_features = len(features_list)
    features_and_values = {}
    with open(txt_path, 'r', encoding='ISO-8859-1') as file, open(csv_path, 'w') as output:
        for line, i in zip(file, range(100000000)):
            # remove commas from text
            line_no_commas = line.replace(',', ' ')
            features_and_values = extract_feature_from_line_to_dict(features_list, line_no_commas, features_and_values)
            if len(features_and_values) == number_of_features:
                written_line = write_dictionary_values_to_string_by_order(features_and_values, features_list)
                output.write(written_line)
                features_and_values = {}
            if i % 10000000 == 0:
                print("this is the {0} iteration".format(i))


def extract_feature_from_line_to_dict(features_list, line, dictionary):
    find_colon = line.find(":")
    feature_name = line[:find_colon]
    if find_colon == -1 or feature_name not in features_list:
        return dictionary
    feature_value = line[find_colon + 1:].strip()
    dictionary[feature_name] = feature_value
    return dictionary


def write_dictionary_values_to_string_by_order(dictionary, ordered_list):
    line = ""
    for item in ordered_list:
        feature_value = dictionary[item] + ','
        line += feature_value
    # remove last ',' to be read by the csv file
    line = line[:-1]
    line += '\n'
    return line


if __name__ == "__main__":
    convert_txt_to_csv("moviesgz.txt", csv_path="movies_score.csv", features_list=["review/text", "review/score"])

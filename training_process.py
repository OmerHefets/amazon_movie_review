from sklearn.feature_extraction.text import HashingVectorizer


def feature_extraction_from_text(text_reviews, n_features):
    reviews_list_of_lists = text_reviews.astype(str).values.tolist()
    # turn the list of lists to one list with strings
    reviews_in_list = [review for sublist in reviews_list_of_lists for review in sublist]
    vectorizer = HashingVectorizer(n_features=n_features)
    text_features = vectorizer.fit_transform(reviews_in_list)
    return text_features


def y_for_ordinal_regression(df, target_value):
    return df.iloc[:, 0].apply(lambda x: 1 if x > target_value else 0)

import split_randomly_datasets
import trials
from sklearn.feature_extraction.text import HashingVectorizer
import time
import OrdinalClassifier as ordin
import AmazonReview as amazon

start = time.time()

# my code here #

# split_randomly_datasets.split_dataset("train_validation.csv", "training_set.csv", "validation_set.csv", 0.75, chunksize=10 ** 6)
# trials.text_preprocessing('validation_set.csv', ['review/text', 'review/score'], 'validation_set_processed.csv')


#process = amazon.AmazonReview('a')
#print(process.ordinal_classifiers.classifiers)


classifier = ordin.OrdinalClassifier()
classifier.partial_fit(data='training_set_processed.csv', n_iter=1, chunks=10 ** 5, features_and_y_list=['review/text', 'review/score'])


# my code here #


end = time.time()
print(end - start)

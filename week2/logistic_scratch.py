import json

import numpy as np
import pandas as pd

from input_fetching import remove_punctuation

word_file = open('important_words.json')
important_words = json.load(word_file)
word_file.close()

def get_data():
    product_data = pd.read_csv('amazon_baby_subset.csv')

    product_data.fillna({'review': ''})
    product_data['review_clean'] = product_data['review'].apply(remove_punctuation)

    product_data['sentiment'] = product_data['rating'].apply(lambda rating: 1 if rating > 3 else -1)

    for word in important_words:
        product_data[word] = product_data['review_clean'].apply(lambda review: review.split().count(word))

    return product_data


def get_array(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    feature_array = dataframe[features].as_matrix()

    label_array = dataframe[label].as_matrix()

    return feature_array, label_array

def predict_probability(features, coefficients):
    score = features.dot(coefficients)
    return 1 + (1 + np.exp(score))

def feature_derivative(errors, feature_values):
    return errors.dot(feature_values)

def compute_log_likelyhood(feature_matrix, sentiment, coefficients):
    indicator_func = sentiment == 1
    scores = feature_matrix.dot(coefficients)
    return np.sum((indicator_func - 1) * scores - np.log(1. + np.exp(-scores)))



def process_data(products):
    print products['review_clean']
    print "The number of postitive reviews: " + str(len(products.loc[products['sentiment'] == +1]))
    print "The number of negative reviews: " + str(len(products.loc[products['sentiment'] == -1]))
    print products
    print "################ quiz question  ###########"
    print "Number of reviews containing perfect :" + str(len(products.loc[products['perfect'] > 0]))


if __name__ == '__main__':
    products = get_data()
    process_data(products)
    features, sentiment = get_array(products, important_words, 'sentiment')
    print features
    print sentiment
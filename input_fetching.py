import pandas as pd
import json
import string
import numpy as np
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def remove_punctuation(text):
    return str(text).translate(None, string.punctuation)


def predict_label(model, vectorizer, data):
    sample_test_matrix = vectorizer.transform(data['review_clean'])
    scores = model.decision_function(sample_test_matrix)
    print "scores: ", scores
    class_predictions = [1 if score > 0 else -1 for score in scores]
    probability_for_positive = 1 / (1 + np.exp( -1 * scores))
    return class_predictions, probability_for_positive

def get_data():
    product_data = pd.read_csv('/home/chris/python_projects/coursera/ml_classification/week_1/amazon_baby.csv')
    file = open('/home/chris/python_projects/coursera/ml_classification/week_1/module-2-assignment-train-idx.json')
    train_indices = json.load(file)
    file.close()
    file = open('/home/chris/python_projects/coursera/ml_classification/week_1/module-2-assignment-test-idx.json')
    test_indices = json.load(file)
    file.close()

    product_data.fillna({'review': ''})
    product_data['review_clean'] = product_data['review'].apply(remove_punctuation)
    non_neutral_data = product_data.loc[product_data.rating != 3]
    non_neutral_data['sentiment'] = non_neutral_data['rating'].apply(lambda rating: 1 if rating > 3 else -1)
    train_data = non_neutral_data.iloc[train_indices]
    test_data = non_neutral_data.iloc[test_indices]
    return train_data, test_data


def build_model(train_data, word_subset=None, read_from_file=False):
    # building the vectorizer that builds the sparse word counter matrix
    sentiment_model = None
    if not read_from_file:
        if not word_subset is None:
            vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', vocabulary=word_subset)
        else:
            vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
        train_matrix = vectorizer.fit_transform(train_data['review_clean'])
        # build the model
        sentiment_model = LogisticRegression()
        sentiment_model.fit(train_matrix, train_data['sentiment'])
        with open('vectorize.pickle', 'wb') as handle:
            pickle.dump(vectorizer, handle)
        with open('model.pickle', 'wb') as handle:
            pickle.dump(sentiment_model, handle)
    else:
        with open('vectorize.pickle', 'rb') as handle:
            vectorizer = pickle.load(handle)
        with open('model.pickle', 'rb') as handle:
            sentiment_model = pickle.load(handle)
    coefficients = sentiment_model.coef_
    print "The number of positive coefficients is " + str(len(coefficients[coefficients > 0]))
    print "The number of negative coefficients is " + str(len(coefficients[coefficients < 0]))
    return sentiment_model, vectorizer

def complex_model(train_data, test_data):
    sentiment_model, vectorizer = build_model(train_data, word_subset=None, read_from_file=True)
    sample_data = test_data.iloc[10:13]
    labels, probabilities = predict_label(sentiment_model, vectorizer, sample_data)
    index = np.argmin(probabilities) + 1
    print "The review with index number " + str(index) + " is the lowest one."
    print labels, probabilities

    # finding the most positive/ negative review
    labels, probabilities = predict_label(sentiment_model, vectorizer, test_data)
    test_data['probability'] = probabilities
    sorted_test_data = test_data.sort_values(['probability'], ascending=[False])
    indices_with_highest_probabilities = sorted_test_data.iloc[:20].index
    print "The indices with the highest review: ", indices_with_highest_probabilities
    print "The ones with the most negative reviews: ", sorted_test_data.iloc[-20:].index

    # determine the accuracy of the model
    test_data['predicted'] = labels
    print "The accuracy of the complex model on the test data is: "
    print determine_accuracy(test_data)

    return sentiment_model, vectorizer


def determine_accuracy(test_data, simple=False):
    column_name = "predicted"
    if simple:
        column_name = column_name + "_simple"
    num_test_observations = len(test_data)
    return float(len(test_data.loc[test_data[column_name] == test_data['sentiment']])) / num_test_observations


def main():
    train_data, test_data = get_data()
    print "train length: ", len(train_data)
    print "test length: ", len(test_data)

    sentiment_model, vectorizer = complex_model(train_data, test_data)

    # build model with significant words
    significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 'work', 'product', 'money', 'would', 'return']
    simple_model, simple_vectorizer = build_model(train_data, significant_words)
    simple_labels, simple_probabilities = predict_label(simple_model, simple_vectorizer, test_data)
    simple_pairs = get_sorted_word_coefficients(simple_vectorizer, simple_model)
    print simple_pairs
    complex_pairs = get_sorted_word_coefficients(vectorizer, sentiment_model)
    for word, coef in simple_pairs:
        for com_word, com_coeff in complex_pairs:
            if word == com_word:
                print word + " has the simple coeff " + str(coef) + " and the complex " + str(com_coeff)


    # accuracy comparison
    labels, _ = predict_label(sentiment_model, vectorizer, train_data)
    train_data['predicted'] = labels
    print "The accuracy of the complex model on the train data is: "
    print determine_accuracy(train_data)

    labels, _ = predict_label(simple_model, simple_vectorizer, train_data)
    train_data['predicted_simple'] = labels
    print "The accuracy of the simple model on the train data is: "
    print determine_accuracy(train_data, simple=True)

    labels, _ = predict_label(sentiment_model, vectorizer, test_data)
    test_data['predicted'] = labels
    print "The accuracy of the complex model on the test data is: "
    print determine_accuracy(test_data)

    labels, _ = predict_label(simple_model, simple_vectorizer, test_data)
    test_data['predicted_simple'] = labels
    print "The accuracy of the simple model on the test data is: "
    print determine_accuracy(test_data, simple=True)


def get_sorted_word_coefficients(vectorizer, model):
    simple_pairs = zip(vectorizer.vocabulary_.keys(), model.coef_[0])
    simple_pairs.sort(key=lambda pair: pair[1])
    return simple_pairs


if __name__ == '__main__':
    main()
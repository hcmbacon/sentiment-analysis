
import pandas as pd
import seaborn as sns
import spacy
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import confusion_matrix
import os, sys


class SentimentAnalysis():
    '''
    Class for training and evaluating neural network for classifying sentiment of movie comments
    '''
    def __init__(self, pos_file, neg_file, model_name='movie_review_classifier.sav'):
        self.pos_file = pos_file
        self.neg_file = neg_file
        self.model_name = model_name
        self.spacy_model = spacy.load('en_core_web_md')

    def get_no_lines(self, text_file):
        '''
        Get the number of lines in text_file
        :param text_file: path as string
        :return no_lines: int
        '''
        with open(text_file) as f:
            no_lines= sum(1 for _ in f)
        f.close()
        return no_lines

    def get_comment_embeddings(self, text_file):
        '''
        Get comment representations for all comments in text_file. For each comment, extracts word embeddings for words in comment
        and averages the word vectors for overall comment embedding.
        :param text_file: path as string
        :return: comment_embeddings: np.array, shape=[no of comments,300]
        '''
        no_lines = self.get_no_lines(text_file)

        # Initialise np.array for storing comment embeddings, where rows are comments and columns are features
        comment_embeddings = np.zeros((no_lines, 300))
        f = open(text_file, 'r')

        # Loop through comments in text_file
        for l, line in enumerate(f):
            tokens = self.spacy_model(unicode(line))
            no_tokens = len(tokens)
            # Initialise np.array for collecting word embeddings from the comment
            word_embeddings = np.zeros((no_tokens, 300))
            for n, t in enumerate(tokens):
                word_embeddings[n] = t.vector
            # Averaging word embeddings and storing as comment representation in comment_embeddings
            comment_embeddings[l] = np.mean(word_embeddings, axis=0)
        f.close()
        return comment_embeddings

    def split_data(self):
        '''
        Assigns labels to comment embeddings (positive=1, negative=0) and splits the data into train_data and test_data.
        Train_data contains 80% of the positive comment embeddings and 80% of the negative comment embeddings.
        Test_data contains 20% of the positive comment embeddings and 80% of the negative comment embeddings.
        Labelled train and test data are shuffled for randomisation in training and are saved in csv files.
        :return: train_data: np.array, shape=[20000,301]
                test_data: np.array, shape=[5000,301]
        '''
        print('Getting comment embeddings for positive movie reviews')
        pos_data = self.get_comment_embeddings(self.pos_file)
        print('Getting comment embeddings for negative movie reviews')
        neg_data = self.get_comment_embeddings(self.neg_file)

        # Adding labels to np.array for positive movie reviews
        labelled_pos_data = np.concatenate((pos_data, np.ones((pos_data.shape[0],1))), axis=1)
        np.random.shuffle(labelled_pos_data)

        # Adding labels to np.array for negative movie reviews
        labelled_neg_data = np.concatenate((neg_data, np.zeros((neg_data.shape[0],1))), axis=1)
        np.random.shuffle(labelled_neg_data)

        # Getting split index
        pos_split_ix = int(pos_data.shape[0]*0.8)
        neg_split_ix = int(neg_data.shape[0]*0.8)

        print('Splitting train and test data')
        train_data = np.concatenate((labelled_pos_data[:pos_split_ix], labelled_neg_data[:neg_split_ix]), axis=0)
        np.random.shuffle(train_data)
        print('Saving training comment embeddings to train_data.csv')
        np.savetxt('train_data.csv',train_data, delimiter=',')
        test_data = np.concatenate((labelled_pos_data[pos_split_ix:], labelled_neg_data[neg_split_ix:]), axis=0)
        np.random.shuffle(test_data)
        print('Saving training comment embeddings to test_data.csv')
        np.savetxt('test_data.csv',test_data, delimiter=',')

        return train_data, test_data

    def get_data_labels(self, data):
        '''
        Extracts comment embeddings and corresponding labels for training.
        :param data: np.array, shape=[n, 301]
        :return: Xval: np.array, shape=[n,300]
                yval: np.array, shape=[n, 1]
        '''
        Xval= data[:,:300]
        yval = data[:,-1]
        return Xval, yval

    def train(self, Xtrain, ytrain):
        '''
        Trains neural network on n-sized Xtrain comment embeddings with ytrain labels for sentiment analysis of comments. Saves model.
        :param Xtrain: np.array, shape=[n, 300]
        :param ytrain: np.array, shape=[n, 1]
        :return: None
        '''
        model = MLPClassifier(early_stopping=True)
        model.fit(Xtrain, ytrain)
        pickle.dump(model, open(self.model_name, 'wb'))

    def evaluate(self, Xtest, ytest):
        '''
        Evaluates saved neural network on n-sized test set. Prints classification accuracy for each class to console.
        :param Xtest: np.array, shape=[n, 300]
        :param ytest: np.array, shape=[n, 1]
        :return: None
        '''
        if os.path.exists(self.model_name):
            model = pickle.load(open(self.model_name, 'rb'))
            y_pred = model.predict(Xtest)
            cm = confusion_matrix(ytest, y_pred)
            print('Classification accuracy for negative reviews: {}'.format(cm[0,0]/float((np.sum(cm[0])))))
            print('Classification accuracy for positive reviews: {}'.format(cm[1,1]/float((np.sum(cm[1])))))
        else:
            print('Model does not exist. Specify "train" as argument')


if __name__ == '__main__':
    mode = sys.argv[1].lower()
    movie_review = SentimentAnalysis('positive_reviews.txt', 'negative_reviews.txt')

    if mode == 'train':
        if os.path.exists('train_data.csv'):
            print('Loading train data features from csv file')
            train_data = np.loadtxt('train_data.csv', delimiter=",")
        else:
            print('Extracting train data features')
            train_data, test_data = movie_review.split_data()
        Xtrain, ytrain = movie_review.get_data_labels(train_data)
        print('Beginning training')
        movie_review.train(Xtrain,ytrain)
        print('Finished training model')

    elif mode == 'evaluate':
        if os.path.exists('test_data.csv'):
            print('Loading test data features from csv file')
            test_data = np.loadtxt('test_data.csv', delimiter=",")
        else:
            print('Extracting test data features')
            train_data, test_data = movie_review.split_data()
        Xtest, ytest = movie_review.get_data_labels(test_data)
        print('Beginning evaluation')
        movie_review.evaluate(Xtest, ytest)
    else:
        print('Please specify "train" or "evaluate" as argument to script')


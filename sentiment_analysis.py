import pandas as pd
import seaborn as sns
import sklearn
import spacy
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
import os, sys



class SentimentAnalysisTrain():
    def __init__(self, pos_file, neg_file):
        self.pos_file = pos_file
        self.neg_file = neg_file

    def get_no_lines(self, text_file):
        with open(text_file) as f:
            no_lines= sum(1 for _ in f)
        f.close()
        return no_lines

    def get_comment_embeddings(self, text_file, spacymodel):
        no_lines = self.get_no_lines(text_file)
        comment_embeddings = np.zeros((no_lines, 300))
        f = open(text_file, 'r')
        for l, line in enumerate(f):
            tokens = spacymodel(unicode(line))
            no_tokens = len(tokens)
            word_embeddings = np.zeros((no_tokens, 300))
            for n, t in enumerate(tokens):
                word_embeddings[n] = t.vector
            comment_embeddings[l] = np.mean(word_embeddings, axis=0)
            # break
        f.close()
        return comment_embeddings

    def split_data(self):
        spacymodel = spacy.load('en_core_web_md')
        pos_data = self.get_comment_embeddings(self.pos_file, spacymodel)
        neg_data = self.get_comment_embeddings(self.neg_file, spacymodel)
        labelled_pos_data = np.concatenate((pos_data, np.ones((pos_data.shape[0],1))), axis=1)
        np.random.shuffle(labelled_pos_data)
        labelled_neg_data = np.concatenate((neg_data, np.zeros((neg_data.shape[0],1))), axis=1)
        np.random.shuffle(labelled_neg_data)
        pos_split_ix = int(pos_data.shape[0]*0.8)
        neg_split_ix = int(neg_data.shape[0]*0.8)
        train_data = np.concatenate((labelled_pos_data[:pos_split_ix], labelled_neg_data[:neg_split_ix]), axis=0)
        np.random.shuffle(train_data)
        np.savetxt('train_data.csv',train_data, delimiter=',')
        test_data = np.concatenate((labelled_pos_data[pos_split_ix:], labelled_neg_data[neg_split_ix:]), axis=0)
        np.random.shuffle(test_data)
        np.savetxt('test_data.csv',test_data, delimiter=',')
        return train_data, test_data


    def get_data_labels(self, data):
        Xval= data[:,:300]
        yval = data[:,-1]
        return Xval, yval

    def train(self, Xtrain, ytrain):
        model = MLPClassifier(early_stopping=True)
        model.fit(Xtrain, ytrain)
        pickle.dump(model, open('movie_review_classifier.sav', 'wb'))


    def evaluate(self, Xtest, ytest):
        if os.path.exists('movie_review_classifier.sav'):
            model = pickle.load(open('movie_review_classifier.sav', 'rb'))
            y_pred = model.predict(Xtest)
            cm = confusion_matrix(ytest, y_pred)
            print('Classification accuracy for negative reviews: {}'.format(cm[0,0]/float((np.sum(cm[0])))))
            print('Classification accuracy for positive reviews: {}'.format(cm[1,1]/float((np.sum(cm[1])))))
        else:
            print('Model does not exist. Specify "train" as argument')


if __name__ == '__main__':
    mode = sys.argv[1].lower()
    movie_review = SentimentAnalysisTrain('positive_reviews.txt', 'negative_reviews.txt')

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


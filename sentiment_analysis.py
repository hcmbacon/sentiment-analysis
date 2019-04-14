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
        print(comment_embeddings.shape)
        return comment_embeddings

    def split_data(self):
        spacymodel = spacy.load('en_core_web_md')
        pos_data = self.get_comment_embeddings(self.pos_file, spacymodel)
        neg_data = self.get_comment_embeddings(self.neg_file, spacymodel)
        labelled_pos_data = np.concatenate((pos_data, np.ones((pos_data.shape[0],1))), axis=1)
        np.random.shuffle(labelled_pos_data)
        labelled_neg_data = np.concatenate((neg_data, np.zeros((neg_data.shape[0],1))), axis=1)
        np.random.shuffle(labelled_neg_data)
        pos_split_ix = int(self.get_no_lines(self.pos_file)*0.8)
        neg_split_ix = int(self.get_no_lines(self.neg_file)*0.8)
        labelled_train = np.concatenate((labelled_pos_data[:pos_split_ix], labelled_neg_data[:neg_split_ix]), axis=0)
        np.random.shuffle(labelled_train)
        np.savetxt('train_data.csv',labelled_train, delimiter=',')
        labelled_test = np.concatenate((labelled_pos_data[pos_split_ix:], labelled_neg_data[neg_split_ix:]), axis=0)
        np.random.shuffle(labelled_test)
        np.savetxt('test_data.csv',labelled_test, delimiter=',')


    def get_train_test_data(self):
        train_data = np.loadtxt('train_data.csv', delimiter=",")
        test_data = np.loadtxt('test_data.csv', delimiter=",")
        Xtrain = train_data[:,:300]
        ytrain = train_data[:,-1]
        Xtest = test_data[:,:300]
        ytest = test_data[:,-1]
        # print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)
        return Xtrain, ytrain, Xtest, ytest

    def train(self, Xtrain, ytrain):
        model = MLPClassifier(early_stopping=True)
        model.fit(Xtrain, ytrain)
        pickle.dump(model, open('movie_review_classifier.sav', 'wb'))


    def evaluate(self):
        self.get_train_test_data()
        model = pickle.load(open('movie_review_classifier.sav', 'rb'))
        y_pred = model.predict(self.Xtest)
        print(confusion_matrix(self.ytest, y_pred))



if __name__ == '__main__':
    movie_review = SentimentAnalysisTrain('/Users/hannahbacon/nlp-challenge/positive_reviews.txt', '/Users/hannahbacon/nlp-challenge/negative_reviews.txt')
    # Xtrain, ytrain, Xtest, ytest = movie_review.split_data()
    Xtrain, ytrain, Xtest, ytest = movie_review.get_train_test_data()
    movie_review.train(Xtrain,ytrain)
    movie_review.evaluate(Xtest, ytest)
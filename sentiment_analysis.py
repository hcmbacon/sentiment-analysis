import pandas as pd
import seaborn as sns
import sklearn
import spacy
import numpy as np
from sklearn.model_selection import train_test_split


# nlp = spacy.load('en_core_web_md')
#
# with open('/Users/hannahbacon/nlp-challenge/negative_reviews.txt') as f:
#     for line in f:
#         tokens = nlp(unicode(line))
#         for t in tokens:
#             if t.is_oov:
#                 print('OOV word: {}'.format(t.text))
#             else:
#                 print(t.lemma_)
#         # text = line.split()
#         # print(text)
#         # for word in text:
#         #     print(word)
#         break
#     f.close()


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
            break
        f.close()
        print(comment_embeddings.shape)
        return comment_embeddings

    def split_data(self):
        spacymodel = spacy.load('en_core_web_md')
        pos_data = self.get_comment_embeddings(self.pos_file, spacymodel)
        neg_data = self.get_comment_embeddings(self.neg_file, spacymodel)




if __name__ == '__main__':
    a = SentimentAnalysisTrain('/Users/hannahbacon/nlp-challenge/positive_reviews.txt', '/Users/hannahbacon/nlp-challenge/negative_reviews.txt')
    a.split_data()
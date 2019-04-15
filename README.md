# Sentiment-analysis

## Classification of sentiment (pos/neg) of movie reviews

Version Requirements:

- Python 2.7.13
- Sci-kit Learn 0.19.0
- spaCy 2.0.13, with downloaded word vector model en_core_web_md. To download model:

    
    python -m spacy download en_core_web_md

Usage for training, from command-line:

    python sentiment_analysis.py train
    
Usage for evaluation, from command-line:

    python sentiment_analysis.py evaluate

## Methodology

For training a neural network to predict the sentiment of movie reviews, the positive_reviews.txt and negative_reviews.txt data were selected for training and testing the model.\
This data provides labelled positive and negative sentiments for examples of movie reviews.\
The unsupervised_reviews.txt data was not used.

### Extracting Comment Embeddings

Each line in the data text files contains a tokenised string of words in a review.\
 300-dimensional word embedding vectors can be extracted for each of these tokens using the pre-trained spacy model.\
 The average of these word embeddings can be calculated for a representation of the entire comment.\
 This averaged word embedding, or comment embedding, contains useful information from the individual words in the review, while normalising for the effect of longer reviews with more words.\
 This representation can also be used for training a feedforward neural network, as it has s a constant vector length as necessary for the input layer.
 These 300-dimensional comment embedding vectors were the input chosen to the neural network.

### Splitting data

The positive (class 1) and negative (class 0) reviews contained 12500 reviews each, for a total of 25000 reviews. \
 The data from each class was randomly shuffled then the first 80% of the positive reviews were combined with the first 80% of the negative reviews, for a total of 20000 examples for the training set.
 The remaining 20% of data from each class was selected for the test set (5000 examples).\
 This split provides a large amount of data for the training for improving generalisation, and the balanced classes provide examples in an even distribution for predicting each class.
 The test set size is also reasonable to give a reliable evaluation for each class.
 Each train and test set were shuffled again to ensure random order of training examples with respect to class.

### Training

A feedforward neural network was trained, using the sci-kit learn toolkit.

The input layer has 300 dimensions. There are 2 hidden layers with 100 dimensions each. RELU activatin function was used and the learning rate was 0.001.

The model is saved as movie_review_classifier.sav.

### Evaluation

The output of the saved model was evaluated on the test set. The classification accuracies were obtained per class.
The movie_review classifier.sav model obtained an accuracy of 83.36% for negative reviews and 87.66% for positive reviews.
This provides a good baseline for further optimisation of the model.




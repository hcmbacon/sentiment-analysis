# Sentiment-analysis

### Classification of sentiment (pos/neg) of movie reviews

Version Requirements:

- Python 2.7.13
- spaCy 2.0.13, with downloaded word vector model en_core_web_md. To download model:

    
    python -m spacy download en_core_web_md

Usage for training, from command-line:

    python sentiment_analysis.py train
    
Usage for evaluation, from command-line:

    python sentiment_analysis.py evaluate

### Methodology

- Choice of datasets

- Splitting the training and testing data

- Word embeddings

- Averaging for sentence length, and same number of features in input layer

- Input Layer

- Neural net parameters

- Evaluation




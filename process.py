import os

import pandas

NLTK_DIR = "./corpus"
INPUT_DIR = "./data"


import nltk

nltk.data.path.append(f"{NLTK_DIR}")
# Define Preprocessing functions (Taken From https://github.com/rohanrao619/Twitter_Sentiment_Analysis)
def tokenize(X):
    """
    Tokenize the data using nltk
    """

    treebank = nltk.tokenize.TreebankWordTokenizer()
    X_tokenized = [treebank.tokenize(sentence) for sentence in X]
    return X_tokenized


def remove_stopwords(X):
    """
    Remove Stopwords using nltk
    """

    stopwords = nltk.corpus.stopwords.words('english') + ['@']
    X_without_stopwords = []

    for sentence in X:
        temp = [word for word in sentence if not word in stopwords]
        X_without_stopwords.append(temp)

    return X_without_stopwords


def stem(X, type='porter'):
    """
    Perform Stemming using nltk
    type = 'Porter','Snowball','Lancaster'
    """

    if type == 'porter':
        stemmer = nltk.stem.PorterStemmer()
    elif type == 'snowball':
        stemmer = nltk.stem.SnowballStemmer()
    elif type == 'lancaster':
        stemmer = nltk.stem.LancasterStemmer()

    X_stemmed = []

    for sentence in X:
        temp = [stemmer.stem(word) for word in sentence]
        X_stemmed.append(temp)

    return X_stemmed


def get_wordnet_pos(treebank_tag):
    """
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
    """
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return 'n'


def lemmatize(X):
    """
    Lemmatize words using corresponding POS tag
    """

    lemmatizer = nltk.stem.WordNetLemmatizer()

    X_pos = []
    X_lemmatized = []

    for sentence in X:
        temp = nltk.pos_tag(sentence)
        X_pos.append(temp)

    for sentence in X_pos:
        temp = [lemmatizer.lemmatize(word[0], pos=get_wordnet_pos(word[1])) for word in sentence]
        X_lemmatized.append(temp)

    return X_lemmatized


def clean_input(X, save=None):

    if not os.path.isfile(save):

        X_tokenized = tokenize(X)

        X_without_stopwords = remove_stopwords ( X_tokenized )

        X_lemmatized = lemmatize(X_without_stopwords)

        X_clean = []
        for sentence in X_lemmatized:
          X_clean.append(" ".join(sentence))


        if save:
            to_save = pandas.DataFrame(X_clean)
            to_save.to_csv(save)
    else:
        print("LOOOAD")
        read = pandas.read_csv(save)
        X_clean = []
        for line in read.iloc[:, 1]:
            X_clean.append(line)


    return X_clean

from keras.preprocessing.text import Tokenizer

import os
import numpy as np
from keras.layers import Embedding

MODEL_DIR = "./model"
EMBEDDING_DIR = "./embeddings"



def create_tokenizer(X_clean, vocab_size=100000):

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_clean)

    return tokenizer


def create_embedding_matrix(vocab_len, embeddings, embedding_size, tokenizer):
    embeddings_index = {}
    f = open(f"{EMBEDDING_DIR}/pretrained/{embeddings}.txt", encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocab_len, embedding_size))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if i < vocab_len:
                embedding_matrix[i] = embedding_vector




    return embedding_matrix

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
import json
import os
import numpy as np
from keras.layers import Embedding

MODEL_DIR = "./model"
VOCAB_DIR = "./vocab"
EMBEDDING_DIR = "./embeddings"

def create_tokenizer(X_clean, vocab_size=100000, save=None):
    tokenizer_path = f"{VOCAB_DIR}/{save}.tokens"

    if os.path.isfile(tokenizer_path):
        with open(tokenizer_path, encoding='latin-1') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
    else:
        oov_token = "<OOV>"
        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
        tokenizer.fit_on_texts(X_clean)

        if save is not None:
            with open(tokenizer_path, 'w', encoding='latin-1') as f:
                f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))

    return tokenizer


def create_embedding_layer(tokenizer, max_length, embeddings, embedding_size, save=None):
    embedding_matrix_path = f"{EMBEDDING_DIR}/matrices/{save}"

    if not os.path.isfile(f"{embedding_matrix_path}.npy"):
        embeddings_index = {}
        f = open(f"{EMBEDDING_DIR}/pretrained/{embeddings}.txt", encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_size))
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        if save:
            np.save(embedding_matrix_path, embedding_matrix)
    else:
        print("Load")
        embedding_matrix = np.load(f"{embedding_matrix_path}.npy")

    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1,
                                output_dim=embedding_size,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)
    return embedding_layer

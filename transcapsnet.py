# CODE FROM https://github.com/XifengGuo/CapsNet-Keras/tree/tf2.2
"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python transcapsnet.py
       python transcapsnet.py --epochs 50
       python transcapsnet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""
import math

import keras.utils

DATA_DIR = "./data"
VOCAB_DIR = "./vocab"
EMBEDDING_DIR = "./embeddings"

import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import logging
import pandas as pd
from process import clean_input
from vocab import create_tokenizer
from vocab import create_embedding_matrix
from keras.utils import pad_sequences
from transformerlayers import Encoder
from keras.preprocessing.text import tokenizer_from_json
import json
from keras.layers import Embedding

K.set_image_data_format('channels_last')

logger = logging.getLogger("LOGGER")
logging.basicConfig(level=logging.INFO)
logger.setLevel(level=logging.INFO)


def CapsNet(input_shape, n_class, routings, batch_size, sequence_length=25, embedding_size=300,  embedding_layer=None, trans_off=False):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param batch_size: size of batch
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape, batch_size=batch_size)


    # Layer 1 in Basic Model, LSTM for feature extraction

    if not trans_off:
        e = Encoder(sequence_length=sequence_length, embedding_layer=embedding_layer, embedding_size=embedding_size, h=8, d_k=64, d_v=64, d_model=128, d_ff=2048, n=1)(x)
    else:
        e = embedding_layer(x)

    lstm = layers.LSTM(512)(e)
    inc_dim = layers.Lambda(lambda x: tf.expand_dims(x, -1))(lstm)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(inc_dim, dim_capsule=8, n_channels=32, kernel_size=32, strides=16, padding='same')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=(input_shape,), name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    # return tf.reduce_mean(tf.square(y_pred))
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))


def train(model,  # type: models.Model
          data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_accuracy',
                                           save_best_only=False, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    # def train_generator(x, y, batch_size, shift_fraction=0.):
    #     train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
    #                                        height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
    #     generator = train_datagen.flow(x, y, batch_size=batch_size)
    #     while 1:
    #         x_batch, y_batch = generator.next()
    #         yield (x_batch, y_batch), (y_batch, x_batch)

    # Training with data augmentation. If shift_fraction=0., n o augmentation.
    model.fit((x_train, y_train), (y_train, x_train),
              steps_per_epoch=int(y_train.shape[0] / args.batch_size),
              epochs=args.epochs,
              validation_data=((x_test, y_test), (y_test, x_test)), batch_size=args.batch_size,
              callbacks=[log, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=64)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    pred_labels = np.argmax(y_pred, 1)
    true_labels = np.argmax(y_test, 1)
    true_p = len([pred_labels[i] for i in range(len(pred_labels)) if pred_labels[i] == 1 and pred_labels[i] == true_labels[i]])
    false_p = len([pred_labels[i] for i in range(len(pred_labels)) if pred_labels[i] == 1 and pred_labels[i] != true_labels[i]])
    true_n = len([pred_labels[i] for i in range(len(pred_labels)) if pred_labels[i] == 0 and pred_labels[i] == true_labels[i]])
    false_n = len([pred_labels[i] for i in range(len(pred_labels)) if pred_labels[i] == 0 and pred_labels[i] != true_labels[i]])

    print(f"Positive Precision: {true_p / (true_p + false_n):.4f}")
    print(f"Negative Precision: {true_n / (true_n + false_p):.4f}")
    print(f"Accuracy: {(true_p + true_n) / (true_p + true_n + false_p + false_n):.4f}")
    print(f"F1: {2 * (true_p / (true_p + false_p + false_n)):.4f}")
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])



def manipulate_latent(model, data, args):
    print('-' * 30 + 'Begin: manipulate' + '-' * 30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_sanders(max_seq, batch_size, tokenizer, testing=True, logger=logger):
    processed_train = f"{DATA_DIR}/sanders/train/train_seq_padded_{max_seq}.csv"
    processed_test = f"{DATA_DIR}/sanders/test/test_seq_padded_{max_seq}.csv"

    if os.path.isfile(processed_train) and os.path.isfile(processed_test):
        logger.info(f"PROCESSED TRAIN DATA ALREADY EXISTS. LOADING: {processed_train}")
        df_train = pd.read_csv(processed_train,
                               index_col=0,
                               encoding="latin-1")

        Y_train = df_train["Sentiment"]
        df_train.drop(columns="Sentiment", inplace=True)
        X_train = df_train.to_numpy()

        logger.info(f"PROCESSED TEST DATA ALREADY EXISTS. LOADING: {processed_test}")
        df_test = pd.read_csv(processed_test,
                              index_col=0,
                              encoding="latin-1")
        Y_test = df_test["Sentiment"]
        df_test.drop(columns="Sentiment", inplace=True)
        X_test = df_test.to_numpy()
    else:
        clean_path = f"{DATA_DIR}/sanders/dataset_cleaned.csv"
        if os.path.isfile(clean_path):
            logger.info(f"CLEAN DATA EXISTS. OPENING: {clean_path}")
            df = pd.read_csv(clean_path, index_col=0, dtype=str, keep_default_na=False)
        else:
            data_path = f"{DATA_DIR}/sanders/sanders.csv"
            logger.info(f"OPENING {data_path}")
            df = pd.read_csv(data_path, names=["Topic",	"Sentiment", "TweetId", "TweetDate", "TweetText"],
                             encoding="latin-1")
            logger.info(f"CLEANING INPUT")
            X_clean = clean_input(df["TweetText"])
            df["TweetText"] = X_clean
            logger.info(f"SAVING CLEANED INPUT TO {clean_path}")
            df.to_csv(clean_path)

        df = df[df.Sentiment.isin(["positive", "negative"])]
        df["Sentiment"] = df["Sentiment"].replace("positive", 1)
        df["Sentiment"] = df["Sentiment"].replace("negative", 0)
        df["Sentiment"] = df["Sentiment"].astype(int)

        test_frac = 0.8
        logger.info(f"CREATING TEST TRAIN SPLIT: Test: {test_frac:.2f} / Train: {1 - test_frac:.2f}")
        df_train = df.sample(frac=test_frac, random_state=42)
        df_test = df.drop(df_train.index)
        df_train.sort_index(inplace=True)
        df_train.reset_index(inplace=True)
        df_test.sort_index(inplace=True)
        df_test.reset_index(inplace=True)

        logger.info(
            f"SAVING TRAIN TEST SPLIT TO {DATA_DIR}/sanders/train/train.csv and {DATA_DIR}/sanders/test/test.csv")
        df_train.to_csv(f"{DATA_DIR}/sanders/train/train.csv")
        df_test.to_csv(f"{DATA_DIR}/sanders/test/test.csv")


        X_train_proc = df_train["TweetText"]
        Y_train = df_train["Sentiment"]

        X_test_proc = df_test["TweetText"]
        Y_test = df_test["Sentiment"]

        X_train_proc = tokenizer.texts_to_sequences(X_train_proc)
        X_test_proc = tokenizer.texts_to_sequences(X_test_proc)

        logger.info("PADDING INPUT")
        padding_type = 'post'
        truncation_type = 'post'

        X_test = pad_sequences(X_test_proc, maxlen=max_seq,
                               padding=padding_type, truncating=truncation_type)

        X_train = pad_sequences(X_train_proc, maxlen=max_seq, padding=padding_type,
                                truncating=truncation_type)


        train_save = pd.concat((pd.DataFrame(Y_train), pd.DataFrame(X_train)), axis=1)
        train_save = train_save.sample(frac=1, random_state=42)
        test_save = pd.concat((pd.DataFrame(Y_test), pd.DataFrame(X_test)), axis=1)
        test_save = test_save.sample(frac=1, random_state=42)
        temp_cols = train_save.columns.to_list()
        temp_cols2 = test_save.columns.to_list()
        temp_cols[0] = "Sentiment"
        temp_cols2[0] = "Sentiment"
        train_save.columns = temp_cols
        test_save.columns = temp_cols2

        train_save.to_csv(processed_train)
        test_save.to_csv(processed_test)

        Y_train = train_save["Sentiment"]
        Y_test = test_save["Sentiment"]

        train_save.drop(columns="Sentiment", inplace=True)
        test_save.drop(columns="Sentiment", inplace=True)

        X_train = train_save.to_numpy()
        X_test = test_save.to_numpy()

    Y_train = to_categorical(tf.constant(Y_train, dtype=tf.float32))
    Y_test = to_categorical(tf.constant(Y_test, dtype=tf.float32))

    num_train_batches = len(X_train) // batch_size
    num_test_batches = len(X_test) // batch_size
    X_train = X_train[:num_train_batches * batch_size]
    Y_train = Y_train[:num_train_batches * batch_size]
    X_test = X_test[:num_test_batches * batch_size]
    Y_test = Y_test[:num_test_batches * batch_size]

    if testing:
        X_test = np.concatenate((X_test, X_train))
        Y_test = np.concatenate((Y_test, Y_train))

    print(len(Y_test))
    return (X_train, Y_train), (X_test, Y_test)

def load_airlines(max_seq, batch_size, tokenizer, testing=True, logger=logger):
    processed_train = f"{DATA_DIR}/airlines/train/train_seq_padded_{max_seq}.csv"
    processed_test = f"{DATA_DIR}/airlines/test/test_seq_padded_{max_seq}.csv"

    if os.path.isfile(processed_train) and os.path.isfile(processed_test):
        logger.info(f"PROCESSED TRAIN DATA ALREADY EXISTS. LOADING: {processed_train}")
        df_train = pd.read_csv(processed_train,
                               index_col=0,
                               encoding="latin-1")

        Y_train = df_train["airline_sentiment"]
        df_train.drop(columns="airline_sentiment", inplace=True)
        X_train = df_train.to_numpy()

        logger.info(f"PROCESSED TEST DATA ALREADY EXISTS. LOADING: {processed_test}")
        df_test = pd.read_csv(processed_test,
                              index_col=0,
                              encoding="latin-1")
        Y_test = df_test["airline_sentiment"]
        df_test.drop(columns="airline_sentiment", inplace=True)
        X_test = df_test.to_numpy()
    else:
        clean_path = f"{DATA_DIR}/airlines/dataset_cleaned.csv"
        if os.path.isfile(clean_path):
            logger.info(f"CLEAN DATA EXISTS. OPENING: {clean_path}")
            df = pd.read_csv(clean_path, index_col=0, dtype=str, keep_default_na=False)
        else:
            data_path = f"{DATA_DIR}/airlines/airlines.csv"
            logger.info(f"OPENING {data_path}")
            df = pd.read_csv(data_path, names=['tweet_id', 'airline_sentiment', 'airline_sentiment_confidence',
       'negativereason', 'negativereason_confidence', 'airline',
       'airline_sentiment_gold', 'name', 'negativereason_gold',
       'retweet_count', 'text', 'tweet_coord', 'tweet_created',
       'tweet_location', 'user_timezone'],
                             encoding="latin-1")
            logger.info(f"CLEANING INPUT")
            X_clean = clean_input(df["text"])
            df["text"] = X_clean
            logger.info(f"SAVING CLEANED INPUT TO {clean_path}")
            df.to_csv(clean_path)

        df = df[df.airline_sentiment.isin(["positive", "negative"])]
        df["airline_sentiment"] = df["airline_sentiment"].replace("positive", 1)
        df["airline_sentiment"] = df["airline_sentiment"].replace("negative", 0)
        df["airline_sentiment"] = df["airline_sentiment"].astype(int)

        test_frac = 0.8
        logger.info(f"CREATING TEST TRAIN SPLIT: Test: {test_frac:.2f} / Train: {1 - test_frac:.2f}")
        df_train = df.sample(frac=test_frac, random_state=42)
        df_test = df.drop(df_train.index)
        df_train.sort_index(inplace=True)
        df_train.reset_index(inplace=True)
        df_test.sort_index(inplace=True)
        df_test.reset_index(inplace=True)

        logger.info(
            f"SAVING TRAIN TEST SPLIT TO {DATA_DIR}/airlines/train/train.csv and {DATA_DIR}/airlines/test/test.csv")
        df_train.to_csv(f"{DATA_DIR}/airlines/train/train.csv")
        df_test.to_csv(f"{DATA_DIR}/airlines/test/test.csv")

        X_train_proc = df_train["text"]
        Y_train = df_train["airline_sentiment"]

        X_test_proc = df_test["text"]
        Y_test = df_test["airline_sentiment"]

        X_train_proc = tokenizer.texts_to_sequences(X_train_proc)
        X_test_proc = tokenizer.texts_to_sequences(X_test_proc)

        logger.info("PADDING INPUT")
        padding_type = 'post'
        truncation_type = 'post'

        X_test = pad_sequences(X_test_proc, maxlen=max_seq,
                               padding=padding_type, truncating=truncation_type)

        X_train = pad_sequences(X_train_proc, maxlen=max_seq, padding=padding_type,
                                truncating=truncation_type)

        train_save = pd.concat((pd.DataFrame(Y_train), pd.DataFrame(X_train)), axis=1)
        train_save = train_save.sample(frac=1, random_state=42)
        test_save = pd.concat((pd.DataFrame(Y_test), pd.DataFrame(X_test)), axis=1)
        test_save = test_save.sample(frac=1, random_state=42)
        temp_cols = train_save.columns.to_list()
        temp_cols2 = test_save.columns.to_list()
        temp_cols[0] = "airline_sentiment"
        temp_cols2[0] = "airline_sentiment"
        train_save.columns = temp_cols
        test_save.columns = temp_cols2

        train_save.to_csv(processed_train)
        test_save.to_csv(processed_test)

        Y_train = train_save["airline_sentiment"]
        Y_test = test_save["airline_sentiment"]

        train_save.drop(columns="airline_sentiment", inplace=True)
        test_save.drop(columns="airline_sentiment", inplace=True)

        X_train = train_save.to_numpy()
        X_test = test_save.to_numpy()

    Y_train = to_categorical(tf.constant(Y_train, dtype=tf.float32))
    Y_test = to_categorical(tf.constant(Y_test, dtype=tf.float32))

    num_train_batches = len(X_train) // batch_size
    num_test_batches = len(X_test) // batch_size
    X_train = X_train[:num_train_batches * batch_size]
    Y_train = Y_train[:num_train_batches * batch_size]
    X_test = X_test[:num_test_batches * batch_size]
    Y_test = Y_test[:num_test_batches * batch_size]

    if testing:
        X_test = np.concatenate((X_test, X_train))
        Y_test = np.concatenate((Y_test, Y_train))

    print(len(Y_test))
    return (X_train, Y_train), (X_test, Y_test)

def load_stocks(max_seq, batch_size, tokenizer, testing=True, logger=logger):
    processed_train = f"{DATA_DIR}/stocks/train/train_seq_padded_{max_seq}.csv"
    processed_test = f"{DATA_DIR}/stocks/test/test_seq_padded_{max_seq}.csv"

    if os.path.isfile(processed_train) and os.path.isfile(processed_test):
        logger.info(f"PROCESSED TRAIN DATA ALREADY EXISTS. LOADING: {processed_train}")
        df_train = pd.read_csv(processed_train,
                               index_col=0,
                               encoding="latin-1")

        Y_train = df_train["Sentiment"]
        df_train.drop(columns="Sentiment", inplace=True)
        X_train = df_train.to_numpy()

        logger.info(f"PROCESSED TEST DATA ALREADY EXISTS. LOADING: {processed_test}")
        df_test = pd.read_csv(processed_test,
                              index_col=0,
                              encoding="latin-1")
        Y_test = df_test["Sentiment"]
        df_test.drop(columns="Sentiment", inplace=True)
        X_test = df_test.to_numpy()
    else:
        clean_path = f"{DATA_DIR}/stocks/dataset_cleaned.csv"
        if os.path.isfile(clean_path):
            logger.info(f"CLEAN DATA EXISTS. OPENING: {clean_path}")
            df = pd.read_csv(clean_path, index_col=0, dtype=str, keep_default_na=False)
        else:
            data_path = f"{DATA_DIR}/stocks/stocks.csv"
            logger.info(f"OPENING {data_path}")
            df = pd.read_csv(data_path, names=['Text', 'Sentiment'],
                             encoding="latin-1")
            logger.info(f"CLEANING INPUT")
            X_clean = clean_input(df["Text"])
            df["Text"] = X_clean
            logger.info(f"SAVING CLEANED INPUT TO {clean_path}")
            df.to_csv(clean_path)

        df = df[df.Sentiment.isin(['-1', '1'])]
        df["Sentiment"] = df["Sentiment"].replace('-1', 0)
        df["Sentiment"] = df["Sentiment"].astype(int)

        test_frac = 0.8
        logger.info(f"CREATING TEST TRAIN SPLIT: Test: {test_frac:.2f} / Train: {1 - test_frac:.2f}")
        df_train = df.sample(frac=test_frac, random_state=42)
        df_test = df.drop(df_train.index)
        df_train.sort_index(inplace=True)
        df_train.reset_index(inplace=True)
        df_test.sort_index(inplace=True)
        df_test.reset_index(inplace=True)

        logger.info(
            f"SAVING TRAIN TEST SPLIT TO {DATA_DIR}/stocks/train/train.csv and {DATA_DIR}/stocks/test/test.csv")
        df_train.to_csv(f"{DATA_DIR}/stocks/train/train.csv")
        df_test.to_csv(f"{DATA_DIR}/stocks/test/test.csv")

        X_train_proc = df_train["Text"]
        Y_train = df_train["Sentiment"]

        X_test_proc = df_test["Text"]
        Y_test = df_test["Sentiment"]

        X_train_proc = tokenizer.texts_to_sequences(X_train_proc)
        X_test_proc = tokenizer.texts_to_sequences(X_test_proc)

        logger.info("PADDING INPUT")
        padding_type = 'post'
        truncation_type = 'post'

        X_test = pad_sequences(X_test_proc, maxlen=max_seq,
                               padding=padding_type, truncating=truncation_type)

        X_train = pad_sequences(X_train_proc, maxlen=max_seq, padding=padding_type,
                                truncating=truncation_type)

        train_save = pd.concat((pd.DataFrame(Y_train), pd.DataFrame(X_train)), axis=1)
        train_save = train_save.sample(frac=1, random_state=42)
        test_save = pd.concat((pd.DataFrame(Y_test), pd.DataFrame(X_test)), axis=1)
        test_save = test_save.sample(frac=1, random_state=42)
        temp_cols = train_save.columns.to_list()
        temp_cols2 = test_save.columns.to_list()
        temp_cols[0] = "Sentiment"
        temp_cols2[0] = "Sentiment"
        train_save.columns = temp_cols
        test_save.columns = temp_cols2

        train_save.to_csv(processed_train)
        test_save.to_csv(processed_test)

        Y_train = train_save["Sentiment"]
        Y_test = test_save["Sentiment"]

        train_save.drop(columns="Sentiment", inplace=True)
        test_save.drop(columns="Sentiment", inplace=True)

        X_train = train_save.to_numpy()
        X_test = test_save.to_numpy()

    Y_train = to_categorical(tf.constant(Y_train, dtype=tf.float32))
    Y_test = to_categorical(tf.constant(Y_test, dtype=tf.float32))

    num_train_batches = len(X_train) // batch_size
    num_test_batches = len(X_test) // batch_size
    X_train = X_train[:num_train_batches * batch_size]
    Y_train = Y_train[:num_train_batches * batch_size]
    X_test = X_test[:num_test_batches * batch_size]
    Y_test = Y_test[:num_test_batches * batch_size]

    if testing:
        X_test = np.concatenate((X_test, X_train))
        Y_test = np.concatenate((Y_test, Y_train))

    print(len(Y_test))
    return (X_train, Y_train), (X_test, Y_test)


def load_sentiment140(max_seq, batch_size, tokenizer, logger=logger):
    processed_train = f"{DATA_DIR}/sentiment140/train/train_seq_padded_{max_seq}.csv"
    processed_test = f"{DATA_DIR}/sentiment140/test/test_seq_padded_{max_seq}.csv"

    if os.path.isfile(processed_train) and os.path.isfile(processed_test):
        logger.info(f"PROCESSED TRAIN DATA ALREADY EXISTS. LOADING: {processed_train}")
        df_train = pd.read_csv(processed_train,
                               index_col=0,
                               encoding="latin-1")

        Y_train = df_train["target"]
        df_train.drop(columns="target", inplace=True)
        X_train = df_train.to_numpy()


        logger.info(f"PROCESSED TEST DATA ALREADY EXISTS. LOADING: {processed_test}")
        df_test = pd.read_csv(processed_test,
                               index_col=0,
                               encoding="latin-1")
        Y_test = df_test["target"]
        df_test.drop(columns="target", inplace=True)
        X_test = df_test.to_numpy()

    else:
        clean_path = f"{DATA_DIR}/{args.dataset}/dataset_cleaned.csv"
        if os.path.isfile(clean_path):
            logger.info(f"CLEAN DATA EXISTS. OPENING: {clean_path}")
            df = pd.read_csv(clean_path, index_col=0, dtype=str, keep_default_na=False)
        else:
            data_path = f"{DATA_DIR}/sentiment140/sentiment140.csv"
            logger.info(f"OPENING {data_path}")
            df = pd.read_csv(data_path, names=['target', 'id', 'date', 'flag', 'user', 'text'],
                             encoding="latin-1")
            logger.info(f"CLEANING INPUT")
            X_clean = clean_input(df["text"])
            df["text"] = X_clean
            logger.info(f"SAVING CLEANED INPUT TO {clean_path}")
            df.to_csv(clean_path)


        df["target"] = df["target"].astype(int)
        test_frac=0.8
        logger.info(f"CREATING TEST TRAIN SPLIT: Test: {test_frac:.2f} / Train: {1 - test_frac:.2f}")
        df_train = df.sample(frac=test_frac, random_state=42)
        df_test = df.drop(df_train.index)
        df_train.sort_index(inplace=True)
        df_train.reset_index(inplace=True)
        df_test.sort_index(inplace=True)
        df_test.reset_index(inplace=True)

        logger.info(f"SAVING TRAIN TEST SPLIT TO {DATA_DIR}/{args.dataset}/train/train.csv and {DATA_DIR}/{args.dataset}/test/test.csv")
        df_train.to_csv(f"{DATA_DIR}/{args.dataset}/train/train.csv")
        df_test.to_csv(f"{DATA_DIR}/{args.dataset}/test/test.csv")

        X_train_proc = df_train["text"]
        Y_train_proc = df_train["target"]

        X_test_proc = df_test["text"]
        Y_test_proc = df_test["target"]

        X_train_proc = tokenizer.texts_to_sequences(X_train_proc)
        X_test_proc = tokenizer.texts_to_sequences(X_test_proc)

        logger.info("PADDING INPUT")
        padding_type = 'post'
        truncation_type = 'post'

        X_test = pad_sequences(X_test_proc, maxlen=max_seq,
                                      padding=padding_type, truncating=truncation_type)

        X_train = pad_sequences(X_train_proc, maxlen=max_seq, padding=padding_type,
                                       truncating=truncation_type)

        logger.info("FORMATTING TARGET DATA")
        Y_train = np.floor_divide(Y_train_proc, 4)
        Y_test = np.floor_divide(Y_test_proc,  4)


        train_save = pd.concat((pd.DataFrame(Y_train), pd.DataFrame(X_train)), axis=1)
        train_save = train_save.sample(frac=1, random_state=42)
        test_save = pd.concat((pd.DataFrame(Y_test), pd.DataFrame(X_test)), axis=1)
        test_save = test_save.sample(frac=1, random_state=42)
        temp_cols = train_save.columns.to_list()
        temp_cols2 = test_save.columns.to_list()
        temp_cols[0] = "target"
        temp_cols2[0] = "target"
        train_save.columns = temp_cols
        test_save.columns = temp_cols2


        train_save.to_csv(processed_train)
        test_save.to_csv(processed_test)

        Y_train = train_save["target"]
        Y_test = test_save["target"]

        train_save.drop(columns="target", inplace=True)
        test_save.drop(columns="target", inplace=True)

        X_train = train_save.to_numpy()
        X_test = test_save.to_numpy()



    Y_train = to_categorical(tf.constant(Y_train, dtype=tf.float32))
    Y_test = to_categorical(tf.constant(Y_test, dtype=tf.float32))




    num_train_batches = len(X_train) // batch_size
    num_test_batches = len(X_test) // batch_size
    X_train = X_train[:num_train_batches * batch_size]
    Y_train = Y_train[:num_train_batches * batch_size]
    X_test = X_test[:num_test_batches * batch_size]
    Y_test = Y_test[:num_test_batches * batch_size]


    return (X_train, Y_train), (X_test, Y_test)

def get_tokenizer(dataset, vocab_size=100000, logger=logger):
    tokenizer_path = f"{VOCAB_DIR}/{dataset}.{vocab_size}.tokens"
    if os.path.isfile(tokenizer_path):
        logger.info(f"TOKENIZER EXISTS. LOADING {tokenizer_path}")
        with open(tokenizer_path, encoding='latin-1') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
    else:
        clean_path = f"{DATA_DIR}/{dataset}/dataset_cleaned.csv"
        if dataset == "sentiment140":
            columns = ['target', 'id', 'date', 'flag', 'user', 'text']
        elif dataset == "sanders":
            columns = ["Topic", "Sentiment", "TweetId", "TweetDate", "TweetText"]

        if os.path.isfile(clean_path):
            logger.info(f"CLEAN DATA EXISTS. OPENING: {clean_path}")
            df = pd.read_csv(clean_path, index_col=0, names=columns, dtype=str, keep_default_na=False)
        else:
            data_path = f"{DATA_DIR}/{dataset}/{dataset}.csv"
            logger.info(f"OPENING {data_path}")

            df = pd.read_csv(data_path, names=columns, encoding="latin-1")
            logger.info(f"CLEANING INPUT")
            X_clean = clean_input(df[columns[-1]])
            df[columns[-1]] = X_clean
            logger.info(f"SAVING CLEANED INPUT TO {clean_path}")
            df.to_csv(clean_path)

        X_clean = df[columns[-1]]
        tokenizer = create_tokenizer(X_clean, vocab_size=vocab_size)

        with open(tokenizer_path, 'w', encoding='latin-1') as f:
            f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))

    return tokenizer

def get_embedding(dataset, embeddings, embedding_size, vocab_len, max_length, tokenizer, logger=logger):
    embedding_matrix_path = f"{EMBEDDING_DIR}/matrices/{dataset}.{embeddings}.{vocab_len}"
    if os.path.isfile(embedding_matrix_path):
        logger.info(f"EMBEDDING MATRIX ALREADY EXISTS. LOADING {embedding_matrix_path}")
        embedding_matrix = np.load(f"{embedding_matrix_path}.npy")
    else:
        logger.info("CREATING EMBEDDING MATRIX")
        embedding_matrix = create_embedding_matrix(vocab_len=vocab_len, embeddings=embeddings, embedding_size=embedding_size, tokenizer=tokenizer)
        np.save(embedding_matrix_path, embedding_matrix)
    logger.info(f"CREATING EMBEDDING LAYER USING {embeddings}")
    embedding_layer = Embedding(input_dim=vocab_len,
                                output_dim=embedding_size,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False)

    return embedding_layer




if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument("--dataset", default="sentiment140", type=str, choices=['sentiment140', 'airlines', 'stocks', 'sanders'], help="Name of Dataset to Load (sentiment140, sanders, or ???)")
    parser.add_argument("--trans_off", default=False, action='store_true', help="Disables the Transformer Layer (Replaced by LSTM)")
    parser.add_argument("--max_seq", default=25, type=int, help="Maximum Number of Words to Consider")
    parser.add_argument("--embeddings", default="glove.6B.100d", type=str, help="Name of Embedding File")
    parser.add_argument("--embedding_size", default=100, type=int, help="Dimensions of Embedding in Embedding File")
    parser.add_argument("--embedding_dataset", default="sentiment140", type=str, help="Dataset Used to Train Model")
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)



    tokenizer = get_tokenizer(args.embedding_dataset)
    if args.dataset == "sentiment140":
        (X_train, Y_train), (X_test, Y_test) = load_sentiment140(args.max_seq, batch_size=args.batch_size, tokenizer=tokenizer)
    elif args.dataset == "sanders":
        (X_train, Y_train), (X_test, Y_test) = load_sanders(args.max_seq, batch_size=args.batch_size, testing=args.testing, tokenizer=tokenizer)
    elif args.dataset == "airlines":
        (X_train, Y_train), (X_test, Y_test) = load_airlines(args.max_seq, batch_size=args.batch_size, testing=args.testing, tokenizer=tokenizer)
    elif args.dataset == "stocks":
        (X_train, Y_train), (X_test, Y_test) = load_stocks(args.max_seq, batch_size=args.batch_size,
                                                             testing=args.testing, tokenizer=tokenizer)
    embedding_layer = get_embedding(dataset=args.embedding_dataset, embeddings=args.embeddings, embedding_size=args.embedding_size, vocab_len=100000, max_length=args.max_seq, tokenizer=tokenizer)




    # define model
    logger.info("CREATING MODEL")
    model, eval_model, manipulate_model = CapsNet(input_shape=(args.max_seq),
                                                  n_class=len(Y_train[0]),
                                                  routings=args.routings,
                                                  sequence_length=args.max_seq,
                                                  embedding_size=args.embedding_size,
                                                  batch_size=args.batch_size,
                                                  embedding_layer=embedding_layer,
                                                  trans_off=args.trans_off)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((X_train, Y_train), (X_test, Y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        # manipulate_latent(manipulate_model, (X_test, Y_test), args)
        test(model=eval_model, data=(X_test, Y_test), args=args)

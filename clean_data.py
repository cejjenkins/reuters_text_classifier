import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer


def setup_split_data(logger, df):
    """Clean unused data and split into test and train."""
    df_selected = df[
        (df["lewis_split"] != "NOT-USED")
        & (df.groupby("topics").topics.transform("count") > 100)
        & (df["topics"] != "")
    ]
    logger.info("Data cleaned")

    train_data = df_selected[df_selected.lewis_split == "TRAIN"].reset_index()
    test_data = df_selected[df_selected.lewis_split == "TEST"].reset_index()
    logger.info("Split into training and testing complete")
    logger.info(
        f"Training data size: {train_data.shape}, test data size {test_data.shape}"
    )

    return train_data, test_data


def map_topics(logger, train_data, test_data):
    """Map the topics to numbers and encode them."""
    mapping = {}
    for x in range(len(set(train_data.topics))):
        mapping[list(set(train_data.topics))[x]] = x

    logger.info(f"There are {len(set(train_data.topics))} lables")

    train_labels = []
    for x in range(len(train_data.topics)):
        temp = mapping[train_data.topics[x]]
        train_labels.append(temp)

    test_labels = []
    for x in range(len(test_data.topics)):
        temp = mapping[test_data.topics[x]]
        test_labels.append(temp)

    logger.info("Training and testing labels mapped.")

    train_labels_encoded = to_categorical(train_labels)
    test_labels_encoded = to_categorical(test_labels)

    return train_labels_encoded, test_labels_encoded


def tokenize_data(logger, train_data, test_data):
    """Assemble and tokenize a corpus."""
    train_data_selected = train_data[["title", "body"]]
    test_data_selected = test_data[["title", "body"]]

    train_data_selected["text"] = train_data_selected["title"].str.cat(
        train_data_selected["body"], sep=" "
    )
    test_data_selected["text"] = test_data_selected["title"].str.cat(
        test_data_selected["body"], sep=" "
    )
    logger.info("Tokenizing corpus.")
    t = Tokenizer(num_words=10000)
    seq = np.concatenate(
        (train_data_selected["text"], test_data_selected["text"]), axis=0
    )
    t.fit_on_texts(seq)
    logger.info(f"Top 10 words in corpus: {list(t.word_counts)[0:10]}")
    xt_train = t.texts_to_matrix(train_data_selected["text"], mode="tfidf")
    xt_test = t.texts_to_matrix(test_data_selected["text"], mode="tfidf")
    logger.info("Tokenizing is complete.")

    return xt_train, xt_test, t


def split_val(logger, xt_train, train_labels):
    """Split validation data from training data."""
    x_val = xt_train[:1000]
    partial_x_train = xt_train[1000:]

    y_val = train_labels[:1000]
    partial_y_train = train_labels[1000:]

    logger.info("Returning train and val set for training.")
    return x_val, partial_x_train, y_val, partial_y_train

import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer


def setup_split_data(df):
    """Clean unused data and split into test and train."""
    df_selected = df[(df['lewis_split'] != 'NOT-USED') & 
    (df.groupby('topics').topics.transform('count')>100) & 
    (df['topics'] != '')]

    train_data = df_selected[df_selected.lewis_split == "TRAIN"].reset_index()
    test_data =  df_selected[df_selected.lewis_split == "TEST"].reset_index()

    return train_data, test_data

def map_topics(train_data, test_data):
    """Map the topics to numbers and encode them."""
    mapping = {}
    for x in range(len(set(train_data.topics))):
        mapping[list(set(train_data.topics))[x]] = x

    train_labels=[]
    for x in range(len(train_data.topics)):
        temp = mapping[train_data.topics[x]]
        train_labels.append(temp)
        
    test_labels=[]
    for x in range(len(test_data.topics)):
        temp = mapping[test_data.topics[x]]
        test_labels.append(temp)

    train_labels_encoded = to_categorical(train_labels)
    test_labels_encoded = to_categorical(test_labels)

    return train_labels_encoded, test_labels_encoded

def tokenize_data(train_data, test_data):
    train_data_selected = train_data[['title', 'body']]
    test_data_selected = test_data[['title', 'body']]

    train_data_selected['text'] = train_data_selected['title'].str.cat(train_data_selected['body'], sep=' ')
    test_data_selected['text'] = test_data_selected['title'].str.cat(test_data_selected['body'], sep=' ')

    t = Tokenizer(num_words=10000)
    seq = np.concatenate((train_data_selected['text'], test_data_selected['text']), axis=0)
    t.fit_on_texts(seq)
    xt_train = t.texts_to_matrix(train_data_selected['text'], mode='tfidf')
    xt_test = t.texts_to_matrix(test_data_selected['text'], mode='tfidf')
    
    return xt_train, xt_test, t
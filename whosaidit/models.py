# from keras.preprocessing.text import Tokenizer
import joblib
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from keras.layers import Dense, Input, Embedding, Bidirectional, GRU, Dropout, Concatenate
from keras.models import Model
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from collections import OrderedDict
import keras
import pandas as pd
import numpy as np
from munch import Munch
from . import utils


data_dir_path = os.path.join(os.path.dirname(__file__), 'data')


#
# NAIVE BAYES MODEL
#


def train_nb_model(data_df, path=None):
    data = get_train_val_test(get_nb_X(data_df), get_y(data_df))
    model = MultinomialNB()
    model.fit(data.X_train, data.y_train)
    return model
    
    
def get_nb_X(data_df):
    counter = CountVectorizer(ngram_range=(1, 3))
    return counter.fit_transform(data_df.lemmas)
    
    
def get_nb_accuracy(model, data_df):
    data = get_train_val_test(get_nb_X(data_df), get_y(data_df))
    y_train_pred = model.predict(data.X_train)
    y_val_pred = model.predict(data.X_val)
    y_test_pred = model.predict(data.X_test)
    scores = Munch(
        train=accuracy_score(data.y_train, y_train_pred),
        val=accuracy_score(data.y_val, y_val_pred),
        test=accuracy_score(data.y_test, y_test_pred)
    )
    return scores


#
# NEURAL NET MODEL
#


def train_nn_model(data_df, file_name=None, max_sequence_length=30, recur_size=64, dense_size=128, epochs=10):
    # Prepare data.
    word_to_vec, word_to_id, embedding_matrix = get_word_maps(data_df)
    X = get_nn_X(data_df, word_to_id)
    y = get_y(data_df)
    data = get_train_val_test(X, y)
    num_classes = np.unique(y).shape[0]
    
    
    # Build the model.
    K.clear_session()
    inputs = Input(shape=(max_sequence_length,))
    embeddings = Embedding(embedding_matrix.shape[0],
                           embedding_matrix.shape[1],
                           weights=[embedding_matrix],
                           trainable=True)(inputs)

    encoded = Bidirectional(GRU(recur_size))(embeddings)
    dense = Dense(dense_size, activation='relu')(encoded)
    dense = Dropout(0.5)(dense)
    outputs = Dense(num_classes, activation='softmax')(dense)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
                  
    print(model.summary())
    
    # Setup checkpointer.
    if file_name is not None:
        utils.archive_data(save_path)
        checkpoint_path = os.path.join(data_dir_path, file_name)
    else:
        checkpoint_path = os.path.join(data_dir_path, 'temp_model.hdf5')
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_acc')
    
    # Train the model.
    model.fit(data.X_train, data.y_train, batch_size=128, epochs=epochs, 
              validation_data=(data.X_val, data.y_val),
              callbacks=[checkpointer])
    
    # y_pred = model.predict(data.X_train)
    # y_pred = np.argmax(y_pred, axis=1)
    # print('Before', accuracy_score(data.y_train, y_pred))
    # 
    # Load the best weights
    model.load_weights(checkpoint_path)
    # model = keras.models.load_model(checkpoint_path)
    
    # y_pred = model.predict(data.X_train)
    # y_pred = np.argmax(y_pred, axis=1)
    # print('After', accuracy_score(data.y_train, y_pred))
    # 
    
    return model, word_to_id
    

def nn_predict_classes(model, X):
    predicted_probs = model.predict(X)
    return np.argmax(predicted_probs, axis=1)


def get_nn_accuracy(model, word_to_id, data_df):
    data = get_train_val_test(get_nn_X(data_df, word_to_id), get_y(data_df))
    y_train_pred = nn_predict_classes(model, data.X_train)
    y_val_pred = nn_predict_classes(model, data.X_val)
    y_test_pred = nn_predict_classes(model, data.X_test)
    
    y_train_pred = nn_predict_classes(model, data.X_train)
    y_val_pred = nn_predict_classes(model, data.X_val)
    y_test_pred = nn_predict_classes(model, data.X_test)

    scores = Munch(
        train=accuracy_score(data.y_train, y_train_pred),
        val=accuracy_score(data.y_val, y_val_pred),
        test=accuracy_score(data.y_test, y_test_pred)
    )
    return scores


def get_word_maps(data_df, min_vocab_size=2000):
    texts = data_df.tokens
    data_vocab = set()
    for token_list in token_strs_to_token_lists(texts):
        for token in token_list:
            data_vocab.add(token)

    word_to_vec = {}
    embeddings_path = os.path.join(data_dir_path, 'glove.6B/glove.6B.200d.txt')
    with open(embeddings_path) as f:
        for line_num, line in enumerate(f):
            values = line.split()
            word = values[0]
            if min_vocab_size < line_num + 1 and word not in data_vocab:
                continue
            if min_vocab_size > line_num + 1:
                continue
            vector = np.asarray(values[1:], dtype='float32')
            word_to_vec[word] = vector
    
    
    total_vocab = data_vocab | set(word_to_vec.keys())
    rand_state = np.random.RandomState(42)
    embedding_matrix = rand_state.rand(len(total_vocab) + 1, 200)
    embedding_matrix[0] = 0
    word_to_id = {'<PAD>': 0}
    for i, word in enumerate(total_vocab):
        word_id = i + 1
        if word in word_to_vec:
            embedding_matrix[word_id] = word_to_vec[word]
        word_to_id[word] = word_id
        
    return word_to_vec, word_to_id, embedding_matrix


def token_strs_to_token_lists(texts):
    to_skip = {'\n', ' '}
    all_tokens = []
    for text in texts:
        tokens = []
        for token in text.split(' '):
            if token not in to_skip:
                tokens.append(token)
        all_tokens.append(tokens)
    return all_tokens


def get_nn_X(data_df, word_to_id, max_sequence_length=30):
    texts = data_df.tokens
    X = np.zeros((len(texts), max_sequence_length), dtype=int)
    
    token_lists = token_strs_to_token_lists(texts)
    texts_ids = []
    for token_list in token_lists:
        texts_ids.append([word_to_id[token] for token in token_list])
    
    for i, text_ids in enumerate(texts_ids):
        text_ids = text_ids[:max_sequence_length]
        X[i, :len(text_ids)] = text_ids
    
    return X


# def get_nn_data(data_df):
#     train_indices, val_indices, test_indices = get_train_val_test_indices(len(nb_X))
#     X = get_nn_X(data_df)
#     y = get_y(data_df)
#     nn_data = Munch(
#         X_train=X[train_indices],
#         X_val=X[val_indices],
#         X_test=X[test_indices],
#         y_train=y[train_indices],
#         y_val=y[val_indices],
#         y_test=y[test_indices],
#     )
#     return nn_data
    

def prepare_dataframe(data, speakers, min_lemmas=3, rename_patterns=None, equal_classes=False, sample_size=None):
    data_df = pd.DataFrame(data)
    
    if rename_patterns is not None:
        for pattern, replacement in rename_patterns:
            data_df.speaker = data_df.speaker.str.replace(pattern, replacement)
    
    # Some of the text speaker names contain parentheticals. E.g.
    # 'Farnsworth (Angrily)'. Here we remove them.
    data_df.speaker = data_df.speaker.str.replace('\s*\(.+\)\s*', '')
    # Remove semi-colons if they were scraped by accidient.
    data_df.speaker = data_df.speaker.str.replace(':', '')
    
    if len(speakers) == 1:
        data_df.loc[data_df.speaker != speakers[0], 'speaker'] = 'Other'
    else:
        data_df = data_df[data_df.speaker.isin(speakers)]
        data_df = data_df[data_df.lemmas.str.count(' ') >= min_lemmas - 1]
        
    if equal_classes:
        if sample_size is None:
            sample_size = data_df.speaker.value_counts()[-1]
        speaker_samples = []
        for speaker in speakers:
            one_speaker_df = data_df[data_df.speaker == speaker]
            speaker_samples.append(one_speaker_df.sample(sample_size))
        data_df = pd.concat(speaker_samples)
    
    return data_df


#
# UTILS
#


def get_y(data_df):
    encoder = LabelEncoder()
    y = encoder.fit_transform(data_df.speaker)
    return y


def get_train_val_test_indices(num_rows, val_ratio=0.2, test_ratio=0.1):
    rand = np.random.RandomState(42)
    indices = rand.permutation(range(num_rows))
    train_ratio = 1 - val_ratio - test_ratio
    train_indices = indices[:int(len(indices) * train_ratio)]
    val_indices = indices[len(train_indices):len(train_indices) + int(len(indices) * val_ratio)]
    test_indices = indices[len(train_indices) + len(val_indices):]
    return train_indices, val_indices, test_indices
    

def get_train_val_test(X, y):
    train_indices, val_indices, test_indices = get_train_val_test_indices(X.shape[0])
    data = Munch(
        X_train=X[train_indices],
        X_val=X[val_indices],
        X_test=X[test_indices],
        y_train=y[train_indices],
        y_val=y[val_indices],
        y_test=y[test_indices],
    )
    return data


    
    




def test():
    # path = os.path.join(data_dir_path, 'futurama_dialogue_elements_with_text_features.pickle')
    # data = joblib.load(path)
    # pd.set_option('expand_frame_repr', False)
    # data_df = prepare_dataframe(
    #     data, 
    #     # ['Fry', 'Bender', 'Leela', 'Farnsworth'], 
    #     ['Bender'],
    #     rename_patterns=[('Professor Farnsworth', 'Farnsworth'), ("Bender's ghost", 'Bender')],
    #     equal_classes=False
    # )
    # model = train_nb_model(data_df)
    # data = get_train_val_test(get_nb_X(data_df), get_y(data_df))
    # print(get_nb_accuracy(model, data_df))
    # print('--')
    # nn_model, word_to_id = train_nn_model(data_df, epochs=10)
    # print(get_nn_accuracy(nn_model, word_to_id, data_df))
    
    path = os.path.join(data_dir_path, 'buffy_dialogue_elements_with_text_features.pickle')
    data = joblib.load(path)
    pd.set_option('expand_frame_repr', False)
    data_df = prepare_dataframe(data, ['Giles'], equal_classes=False, min_lemmas=4)
    
    model = train_nb_model(data_df)
    data = get_train_val_test(get_nb_X(data_df), get_y(data_df))
    print(get_nb_accuracy(model, data_df))
    print('--')
    nn_model, word_to_id = train_nn_model(data_df, epochs=10)
    print(get_nn_accuracy(nn_model, word_to_id, data_df))
    
    
    # word_to_vec, word_to_id, embedding_matrix = get_word_maps(df.tokens)
    # nn_X = prepare_nn_X(df.tokens, word_to_id)
    # encoder = LabelEncoder()
    # y = encoder.fit_transform(df.speaker)
    # train_indices, val_indices, test_indices = get_train_val_test_indices(len(nn_X))
    # 
    # y_train = y[train_indices]
    # y_val = y[val_indices]
    # y_test = y[test_indices]
    # 
    # nn_X_train = nn_X[train_indices]
    # nn_X_val = nn_X[val_indices]
    # nn_X_test = nn_X[test_indices]
    # 
    # print(embedding_matrix.shape)
    # 
    # model = get_nn_model(embedding_matrix, 2)
    # print(model.summary())
    # model.fit(nn_X_train, y_train, batch_size=128, epochs=10, 
    #           validation_data=(nn_X_val, y_val))
    #           callbacks=[checkpointer])


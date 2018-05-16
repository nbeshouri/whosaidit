
import joblib
import os
import re
from collections import OrderedDict
import pandas as pd
import numpy as np
from munch import Munch
from functools import partial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, precision_score, recall_score
from . import utils


data_dir_path = os.path.join(os.path.dirname(__file__), 'data')


#
# NAIVE BAYES MODEL
#


def train_nb_model(data):
    model = make_pipeline(
        CountVectorizer(ngram_range=(1, 3)),
        MultinomialNB()
    )
    model.fit(data.X_train, data.y_train)    
    return model
    

def save_nb_model(model, file_name):
    utils.archive_data(file_name)
    path = os.path.join(data_dir_path, file_name)
    joblib.dump(model, path)


def load_nb_model(file_name):
    path = os.path.join(data_dir_path, file_name)
    return joblib.load(path)
    

def get_nb_metrics(model, data, **kwargs):
    args = [
        model.predict(data.X_train), 
        model.predict(data.X_val), 
        model.predict(data.X_test), 
        data
    ]
    return get_metrics(*args, **kwargs)


#
# NEURAL NET MODEL
#


def train_nn_model(data, embedding_matrix, file_name=None, max_sequence_length=30, 
                   recur_size=128, dense_size=128, epochs=10):
    # Do this import conditionally because they're slow and not all
    # uses of this module requires keras.
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint
    from keras.layers import (
        Dense, Input, 
        Embedding, 
        Bidirectional, 
        GRU, 
        Dropout, 
        Concatenate
    )
    
    num_classes = np.unique(data.y_train).shape[0]
    
    inputs = Input(shape=(max_sequence_length,))
    embeddings = Embedding(embedding_matrix.shape[0],
                           embedding_matrix.shape[1],
                           weights=[embedding_matrix],
                           trainable=True)(inputs)

    encoded = Bidirectional(GRU(recur_size, dropout=0.5))(embeddings)
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
    
    model.load_weights(checkpoint_path)
    
    return model
    

def nn_predict_classes(model, X):
    predicted_probs = model.predict(X)
    return np.argmax(predicted_probs, axis=1)


def get_nn_metrics(model, data, word_to_id, **kwargs):
    args = [
        nn_predict_classes(model, data.X_train),
        nn_predict_classes(model, data.X_val),
        nn_predict_classes(model, data.X_test),
        data
    ]
    return get_metrics(*args, **kwargs) 


def get_word_maps(texts, min_vocab_size=0):
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


def get_nn_X(texts, word_to_id, max_sequence_length=30):
    X = np.zeros((len(texts), max_sequence_length), dtype=int)
    
    token_lists = token_strs_to_token_lists(texts)
    texts_ids = []
    for token_list in token_lists:
        word_ids = []
        for token in token_list:
            if token in word_to_id:
                word_ids.append(word_to_id[token])
        texts_ids.append(word_ids)
                
    for i, text_ids in enumerate(texts_ids):
        text_ids = text_ids[:max_sequence_length]
        X[i, :len(text_ids)] = text_ids

    return X
    

def prepare_dataframe(data, speakers, min_lemmas=3, rename_patterns=None, 
                      equal_classes=False, sample_size=None):
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
    elif len(speakers) > 1:
        data_df = data_df[data_df.speaker.isin(speakers)]
    
    # Filter for the lemma count.
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

    
def save_nn_model(model, word_to_id, model_file_name):
    import keras
    utils.archive_data(model_file_name)
    model_path = os.path.join(data_dir_path, model_file_name)
    model.save(model_path)
    
    word_to_id_file_name = model_file_name.split('.')[0]
    word_to_id_file_name += '_word_to_id.pickle'
    utils.archive_data(word_to_id_file_name)
    word_to_id_path = os.path.join(data_dir_path, word_to_id_file_name)
    joblib.dump(word_to_id, word_to_id_path)


def load_nn_model(model_file_name):
    import keras
    model_path = os.path.join(data_dir_path, model_file_name)
    model = keras.models.load_model(model_path)
    
    word_to_id_file_name = model_file_name.split('.')[0]
    word_to_id_file_name += '_word_to_id.pickle'
    word_to_id_path = os.path.join(data_dir_path, word_to_id_file_name)
    word_to_id = joblib.load(word_to_id_path)
    return model, word_to_id


#
# UTILS
#


def get_y(data_df):
    encoder = LabelEncoder()
    y = encoder.fit_transform(data_df.speaker)
    return y


def get_train_val_test_indices(num_rows, val_ratio=0.15, test_ratio=0.15):
    rand = np.random.RandomState(42)
    indices = rand.permutation(range(num_rows))
    train_ratio = 1 - val_ratio - test_ratio
    train_indices = indices[:int(len(indices) * train_ratio)]
    val_indices = indices[len(train_indices):len(train_indices) + int(len(indices) * val_ratio)]
    test_indices = indices[len(train_indices) + len(val_indices):]
    return train_indices, val_indices, test_indices
    

def get_train_val_test(X, y, oversample=True):
    # Convert to standard np.array to avoid any indexing oddess
    # incase the input is a Series.
    X = np.array(X)
    
    train_indices, val_indices, test_indices = get_train_val_test_indices(X.shape[0])
    data = Munch(
        X_train=X[train_indices],
        X_val=X[val_indices],
        X_test=X[test_indices],
        y_train=y[train_indices],
        y_val=y[val_indices],
        y_test=y[test_indices],
    )
    
    if oversample:
        # Pandas needs X to be a flat list of things and the data
        # for the NN model isn't. Just putting each row in Python
        # list seems to make it happy. 
        have_non_flat_X = len(X.shape) > 1
        if have_non_flat_X:
            data.X_train = [x for x in data.X_train]
        
        df = pd.DataFrame({'X': data.X_train, 'y': data.y_train})
        counts = df.y.value_counts()
        majority_class = counts.index[0]
        other_classes = counts.index[1:]
        samples = [df]
        for cur_class in other_classes:
            num_to_sample = counts[majority_class] - counts[cur_class]
            just_cur_class_df = df[df.y == cur_class]
            samples.append(just_cur_class_df.sample(num_to_sample, replace=True, random_state=42))
        
        new_df = pd.concat(samples)
        new_df = new_df.sample(len(new_df), random_state=42)
        
        data.X_train = new_df.X.as_matrix()
        data.y_train = new_df.y.as_matrix()
        
        # Converting a Series of ndarrays to a matrix produces
        # a ndarray of ndarrays which has a flat shape. Here we
        # convert them back into a single 2D ndarray.
        if have_non_flat_X:
            data.X_train = np.vstack(data.X_train)
    
    return data
    
    

def get_metrics(y_train_pred, y_val_pred, y_test_pred, 
                data, average='weighted', pos_label=None):
    
    kwargs = dict(average=average, pos_label=pos_label)
    precision_s = partial(precision_score, **kwargs)
    recall_s = partial(recall_score, **kwargs)
    f1_s = partial(f1_score, **kwargs)
    
    scores = Munch(
        train_accuracy=accuracy_score(data.y_train, y_train_pred),
        val_accuracy=accuracy_score(data.y_val, y_val_pred),
        test_accuracy=accuracy_score(data.y_test, y_test_pred),
        
        train_precision=precision_s(data.y_train, y_train_pred),
        val_precision=precision_s(data.y_val, y_val_pred),
        test_precision=precision_s(data.y_test, y_test_pred),
        
        train_recall=recall_s(data.y_train, y_train_pred),
        val_recall=recall_s(data.y_val, y_val_pred),
        test_recall=recall_s(data.y_test, y_test_pred),
        
        train_f1=f1_s(data.y_train, y_train_pred),
        val_f1=f1_s(data.y_val, y_val_pred),
        test_f1=f1_s(data.y_test, y_test_pred),
    )
    return scores
    

# def test():
#     path = os.path.join(data_dir_path, 'buffy_dialogue_elements_with_text_features.pickle')
#     data = joblib.load(path)
#     pd.set_option('expand_frame_repr', False)
#     data_df = prepare_dataframe(
#         data,
# 
#         ['Buffy', 'Willow', 'Xander', 'Giles', 'Spike'],
#         min_lemmas=4
#     )
# 
#     pre_fix = 'buffy_main_cast'
# 
#     # Naive Bayes
#     # data = get_train_val_test(data_df.lemmas, get_y(data_df))
#     # model = train_nb_model(data)
#     # save_nb_model(model, f'{pre_fix}_nb_model.pickle')
#     # print(get_nb_metrics(model, data, average='weighted'))
# 
#     # NN
#     word_to_vec, word_to_id, embedding_matrix = get_word_maps(data_df.tokens)
#     X = get_nn_X(data_df.tokens, word_to_id)
#     y = get_y(data_df)
#     data = get_train_val_test(X, y)
#     nn_model = train_nn_model(data, embedding_matrix, epochs=20)
#     save_nn_model(nn_model, word_to_id, f'{pre_fix}_nn_model.hdf5')
# 
#     print(get_nn_metrics(nn_model, data, word_to_id, average='weighted'))


def train_demo_model():
    path = os.path.join(data_dir_path, 'futurama_dialogue_elements_with_text_features.pickle')
    data = joblib.load(path)
    pd.set_option('expand_frame_repr', False)
    data_df = prepare_dataframe(
        data,

        ['Fry', 'Bender', 'Leela', 'Farnsworth', 'Zoidberg'],
        rename_patterns=[('Professor Farnsworth', 'Farnsworth'), ("Bender's ghost", 'Bender')],
        equal_classes=False,
        min_lemmas=3
    )

    pre_fix = 'futurama_demo_no_punct'

    # Naive Bayes
    data = get_train_val_test(data_df.lemmas, get_y(data_df))
    model = train_nb_model(data)
    save_nb_model(model, f'{pre_fix}_nb_model.pickle')
    print(get_nb_metrics(model, data, average='weighted'))

    # NN
    word_to_vec, word_to_id, embedding_matrix = get_word_maps(texts, min_vocab_size=65000)
    X = get_nn_X(texts, word_to_id)
    y = get_y(data_df)
    data = get_train_val_test(X, y)
    nn_model = train_nn_model(data, embedding_matrix, epochs=20)
    save_nn_model(nn_model, word_to_id, f'{pre_fix}_nn_model.hdf5')
    print(get_nn_metrics(nn_model, data, word_to_id, average='weighted'))


def test():
    path = os.path.join(data_dir_path, 'futurama_dialogue_elements_with_text_features.pickle')
    data = joblib.load(path)
    pd.set_option('expand_frame_repr', False)
    data_df = prepare_dataframe(
        data,

        ['Fry', 'Bender', 'Leela', 'Farnsworth', 'Zoidberg'],
        rename_patterns=[('Professor Farnsworth', 'Farnsworth'), ("Bender's ghost", 'Bender')],
        equal_classes=False,
        min_lemmas=3
    )

    pre_fix = 'futurama_demo_no_punct'

    # Naive Bayes
    data = get_train_val_test(data_df.lemmas, get_y(data_df))
    model = train_nb_model(data)
    save_nb_model(model, f'{pre_fix}_nb_model.pickle')
    print(get_nb_metrics(model, data, average='weighted'))

    # NN
    # print(texts)
    word_to_vec, word_to_id, embedding_matrix = get_word_maps(texts, min_vocab_size=65000)
    X = get_nn_X(texts, word_to_id)
    y = get_y(data_df)
    data = get_train_val_test(X, y)
    nn_model = train_nn_model(data, embedding_matrix, epochs=20)
    save_nn_model(nn_model, word_to_id, f'{pre_fix}_nn_model.hdf5')
    print(get_nn_metrics(nn_model, data, word_to_id, average='weighted'))


# def test():
#     path = os.path.join(data_dir_path, 'futurama_dialogue_elements_with_text_features.pickle')
#     data = joblib.load(path)
#     pd.set_option('expand_frame_repr', False)
#     data_df = prepare_dataframe(
#         data, 
#         ['Fry', 'Bender'],
#         rename_patterns=[('Professor Farnsworth', 'Farnsworth'), ("Bender's ghost", 'Bender')],
#         equal_classes=False,
#         min_lemmas=3
#     )
# 
#     pre_fix = 'bender_vs_fry'
# 
#     # Naive Bayes
#     data = get_train_val_test(data_df.lemmas, get_y(data_df))
#     model = train_nb_model(data)
#     save_nb_model(model, f'{pre_fix}_nb_model.pickle')
#     print(get_nb_metrics(model, data, average='binary'))
# 
#     # NN
#     word_to_vec, word_to_id, embedding_matrix = get_word_maps(data_df.tokens)
#     X = get_nn_X(data_df.tokens, word_to_id)
#     y = get_y(data_df)
#     data = get_train_val_test(X, y)
#     nn_model = train_nn_model(data, embedding_matrix, epochs=20)
#     save_nn_model(nn_model, word_to_id, f'{pre_fix}_nn_model.hdf5')
# 
#     print(get_nn_metrics(nn_model, data, word_to_id, average='weighted'))
    
# 

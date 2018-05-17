"""
This module contains the code to train both of the two models as well
as some model specific and show specific utilities.

Todo:
    * Some of these functions might be useful elsewhere and should
        be moved to the utils or transforms modules.
    * With two models and two shows the name space is getting cluttered.
        If you expand this, the module should probably be broken up.
    * Use logging module to replace print statements.

"""

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
# MODEL TRAINERS
#


def train_bender_vs_fry_models():
    """Train the models that recognize Bender and Fry."""
    data_df = get_futurama_dataframe(['Fry', 'Bender'])
    return train_models(data_df, 'bender_vs_fry', min_vocab_size=0)


def train_futurama_main_cast_models():
    """Train the model that recognize the 5 main characters in Futurama."""
    main_cast = ['Fry', 'Bender', 'Leela', 'Farnsworth', 'Zoidberg']
    data_df = get_futurama_dataframe(main_cast)
    return train_models(data_df, 'futurama_main_cast', min_vocab_size=0)


def train_buffy_main_cast_models():
    """Train the model that recognize the 5 main characters in Buffy."""
    main_cast = ['Buffy', 'Willow', 'Xander', 'Spike', 'Giles']
    data_df = get_buffy_dataframe(main_cast)
    return train_models(data_df, 'buffy_main_cast', min_vocab_size=0)


def train_futurama_cli_models():
    """
    Train a version of the Futurama main cast model with an extended
    vocabulary for use in the command line interface.
    """
    main_cast = ['Fry', 'Bender', 'Leela', 'Farnsworth', 'Zoidberg']
    data_df = get_futurama_dataframe(main_cast)
    return train_models(data_df, 'futurama_main_cast', min_vocab_size=65000)
    

def dry_run():
    """Test the model training pipeline without writing anything to disk."""
    # Test Futurama.
    main_cast = ['Fry', 'Bender', 'Leela', 'Farnsworth', 'Zoidberg']
    data_df = get_futurama_dataframe(main_cast)
    train_models(data_df, None, epochs=1)
    # Test Buffy.
    main_cast = ['Buffy', 'Willow', 'Xander', 'Spike', 'Giles']
    data_df = get_buffy_dataframe(main_cast)
    train_models(data_df, None, epochs=1)


def train_models(data_df, file_prefix=None, epochs=20, min_vocab_size=0, oversample=True):
    """
    Train and return both models based on the characters in `data_df`.
    
    Args:
        data_df (DataFrame): A `DataFrame` containing rows for only the
            characters you want the returned models to classify.
        file_prefix (Optional[str]): The file prefix that will appended to saved
            models. If not given, the models will not be saved.
        epochs (Optional[int]): The number of epochs train the NN model.
            Defaults to 10.
        min_vocab_size (Optional[int]): The minimum number words to add
            to the embedding matrix that are not in the character
            dialogue. All the words the characters say will always be
            added.
        oversample (Optional[bool]): Whether or not oversample the
            minority classes to achieve balanced classes in the training
            set.
        
    Returns:
        (tuple): tuple containing:

            nb_model: An sklearn pipeline containing a `CountVectorizer`
                and a `MultinomialNB` model.
            nn_model: A `keras` model.
            word_to_id (dict): A dictionary mapping word tokens to ids
                in the embedding space.
        
    """
    
    # Naive Bayes
    data = get_train_val_test(data_df.lemmas, get_y(data_df.speaker), oversample)
    nb_model = train_nb_model(data)
    if file_prefix is not None:
        save_nb_model(model, f'{file_prefix}_nb_model.pickle')
    print('Naive Bayes metrics:\n')
    print(get_nb_metrics(nb_model, data, average='weighted'))

    # NN
    word_to_vec, word_to_id, embedding_matrix = get_embeddings(data_df.tokens, min_vocab_size=min_vocab_size)
    X = get_nn_X(data_df.tokens, word_to_id)
    y = get_y(data_df.speaker)
    data = get_train_val_test(X, y, oversample)
    nn_model = train_nn_model(data, embedding_matrix, epochs=epochs)
    if file_prefix is not None:
        save_nn_model(nn_model, word_to_id, f'{file_prefix}_nn_model.hdf5')
    print('Neural net metrics:\n')
    print(get_nn_metrics(nn_model, data, word_to_id, average='weighted'))
    
    return nb_model, nn_model, word_to_id


#
# NAIVE BAYES MODEL
#


def train_nb_model(data):
    """Train and return a naive Bayes model from a data `Munch`."""
    model = make_pipeline(
        CountVectorizer(ngram_range=(1, 3)),
        MultinomialNB()
    )
    model.fit(data.X_train, data.y_train)    
    return model
    

def save_nb_model(model, file_name):
    """Save a naive Bayes model to the data folder."""
    utils.archive_data(file_name)
    path = os.path.join(data_dir_path, file_name)
    joblib.dump(model, path)


def load_nb_model(file_name):
    """Load a naive Bayes model to the data folder."""
    path = os.path.join(data_dir_path, file_name)
    return joblib.load(path)
    

def get_nb_metrics(model, data, **kwargs):
    """Return a `Series` of classification metrics for `model`."""
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
    """
    Train and return a recurrent neural network classifier on `data`.
    
    Args:
        data (Munch): A Munch object containing the train, val, and test
            data. Note that the `X` in should already have been converted
            to embedding indicies.
        embedding_matrix (np.ndarray): A numpy array that maps embedding
            indicies to embedding vectors. This is used to initialize
            the embedding weights.
        file_name (Optional[str]): The name of the file in the data folder 
            to store the model checkpoints. If not given, the a temporary 
            file will be used instead.
        max_sequence_length (Optional[int]): The sequence length that 
            input batches will have.
        recur_size (Optional[int]): The number of units in the GRU cells.
        dense_size (Optional[int]): The number of unitls in the dense hidden layer.
        epochs (Optional[int]): The number of epochs to train the model.
    
    Returns:
        A trained `keras` model.
    
    """
    
    # Do this import conditionally because they're slow and not all
    # uses of this module requires keras.
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint
    from keras.layers import (
        Dense, 
        Input, 
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
    
    # Load the best weights.
    model.load_weights(checkpoint_path)
    
    return model
    

def nn_predict_classes(model, X):
    """Use `model` to predict class ids"""
    predicted_probs = model.predict(X)
    return np.argmax(predicted_probs, axis=1)


def get_nn_metrics(model, data, word_to_id, **kwargs):
    """Return a `Series` of classification metrics for `model`."""
    args = [
        nn_predict_classes(model, data.X_train),
        nn_predict_classes(model, data.X_val),
        nn_predict_classes(model, data.X_test),
        data
    ]
    return get_metrics(*args, **kwargs) 


def get_embeddings(texts, min_vocab_size=0):
    """
    Args:
        texts (Iterable[str]): A sequence of text to fit the embeddings
            on.
        min_vocab_size (Optional[int]): Minimum number of words, not
            counting those in `texts` to include in the embedding
            vocabulary.
    Returns:
        (tuple): tuple containing:
            word_to_vec (dict): A map between word tokens and numpy vectors.
            word_to_id (dict): A map between word tokens and embedding ids.
            embedding_matrix (np.ndarry): A numpy matrix that maps between
                embedding ids and embedding vectors.
    
    """
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
    """Convert a list of strings to a list of token lists."""
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
    """
    Args:
        texts (Iterable[str]): A sequence of texts to fit the embeddings
            on.
        word_to_id (dict): A map between word tokens and embedding ids.
        max_sequence_length (Optional[int]): The maximum number of words
            to include in each line of dialogue. Shorter sequences will
            be padded with the <PAD> vector.
    
    Returns:
        X (np.ndarray): An array with shape `(len(texts), max_sequence_length)`
            containing the correct embedding ids.
    
    """
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
# MISC
#


def prepare_dataframe(data, speakers, min_lemmas=3, rename_patterns=None):
    """
    Selectively convert the raw data into a `DataFrame`
    
    Args:
        data (Iterable[int]): A list of dictionaries that each 
            represent lines of dialogue.
        speakers (Iterable[str]): The characters to include in the
            returned `DataFrame`. If only one speaker is given then all
            the other speakers are renamed 'Other' and all lines of 
            dialogue are returned. This is used for doing one vs. all 
            comparisons.
        min_lemmas (Optional[int]): The minimum number of lemmas required
            for a line of dialogue to be included in the returned 
            `DataFrame`.
        rename_patterns (Optional[Iterable[Tuple[str, str]]]): A set of replacements
            in the form `(pattern, replacement)` to perform on speaker
            names.
    
    Returns:
        (Pandas.DataFrame): A `DataFrame` containing just the lines
            of the requested characters.
    """
    
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
    
    return data_df


def get_y(speaker_list):
    """
    Encode a list of speaking as a list of ints.
    
    Args:
        speaker_lists (List[str]): The list of speaker names.
        
    Returns:
        (List[int]): The speaker names transformed into speaker codes.
            These are assigned alphabetically.
    
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(speaker_list)
    return y


def get_train_val_test_indices(num_rows, val_ratio=0.15, test_ratio=0.15):
    """Return indices of the train, test, and validation sets."""
    rand = np.random.RandomState(42)
    indices = rand.permutation(range(num_rows))
    train_ratio = 1 - val_ratio - test_ratio
    train_indices = indices[:int(len(indices) * train_ratio)]
    val_indices = indices[len(train_indices):len(train_indices) + int(len(indices) * val_ratio)]
    test_indices = indices[len(train_indices) + len(val_indices):]
    return train_indices, val_indices, test_indices
    

def get_train_val_test(X, y, oversample=True):
    """
    Split `X` and `y` into train/test/val sets and return the results
    as a `Munch`. 
    
    Note: 
        I roll my oversampling routine here because `imblearn` didn't
        some of my Xs and ys and I didn't wnat to change my whole 
        pipeline.
        
    """
    # Convert to standard np.array to avoid any indexing oddess
    # in case X is a Series or DataFrame.
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
        # for the NN model isn't. Just putting each row in a Python
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
    """Returns a Series of classification metrics."""
    
    kwargs = dict(average=average, pos_label=pos_label)
    precision_s = partial(precision_score, **kwargs)
    recall_s = partial(recall_score, **kwargs)
    f1_s = partial(f1_score, **kwargs)
    
    scores = {
        'Train accuracy': accuracy_score(data.y_train, y_train_pred),
        'Validation accuracy': accuracy_score(data.y_val, y_val_pred),
        'Test accuracy': accuracy_score(data.y_test, y_test_pred),
        
        'Train precision': precision_s(data.y_train, y_train_pred),
        'Validation precision': precision_s(data.y_val, y_val_pred),
        'Test precision': precision_s(data.y_test, y_test_pred),
        
        'Train recall': recall_s(data.y_train, y_train_pred),
        'Validation recall': recall_s(data.y_val, y_val_pred),
        'Test recall': recall_s(data.y_test, y_test_pred),
        
        'Train F1': f1_s(data.y_train, y_train_pred),
        'Val F1': f1_s(data.y_val, y_val_pred),
        'Test F1': f1_s(data.y_test, y_test_pred),
    }
    
    return pd.Series(scores)


def get_futurama_dataframe(characters):
    """Return a `DataFrame` with lines from Futurama characters."""
    path = os.path.join(data_dir_path, 'futurama_dialogue_elements_with_text_features.pickle')
    data = joblib.load(path)
    data_df = prepare_dataframe(
        data,
        characters,
        rename_patterns=[('Professor Farnsworth', 'Farnsworth'), ("Bender's ghost", 'Bender')],
        min_lemmas=3
    )
    return data_df


def get_buffy_dataframe(characters):
    """Return a `DataFrame` with lines from Buffy characters."""
    path = os.path.join(data_dir_path, 'buffy_dialogue_elements_with_text_features.pickle')
    data = joblib.load(path)
    data_df = prepare_dataframe(data, characters, min_lemmas=4)
    return data_df

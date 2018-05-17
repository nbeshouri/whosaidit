"""
This module contains the the tool's command line interface.

"""

import os
import sys
from contextlib import contextmanager
import click


@click.command()
def cli():
    print('Loading models...')
    # Do some slow imports.
    import pandas as pd
    from . import models
    from . import transforms
    with suppress_stderr():
        import keras
    
    pre_fix = 'futurama_demo'
    
    # Load models.
    nb_model = models.load_nb_model(f'{pre_fix}_nb_model.pickle')
    nn_model, word_to_id = models.load_nn_model(f'{pre_fix}_nn_model.hdf5')
    class_labels = sorted(['Fry', 'Bender', 'Leela', 'Farnsworth', 'Zoidberg'])
    print('Models loaded.')
    
    while True:
        
        text = click.prompt('Please enter some dialogue (or q to quit)', type=str)
        if text == 'q' or text == 'quit':
            return
            
        normed_text = transforms.normalize(text)
        lemmas = transforms.to_lemmas(normed_text)
        tokens = transforms.to_tokens(normed_text)
        
        # Show text tokens and lemmas.
        print(f'\nInput: "{text}"')
        print(f'Tokens: "{tokens}"')
        print(f'Lemmas: "{lemmas}"')
        
        # Compute and print character probs with naive Bayes. 
        nb_pred_probs = nb_model.predict_proba([lemmas])[0]
        nb_results = pd.Series(nb_pred_probs, index=class_labels)
        nb_results = nb_results.map(lambda x: f'{x:.3%}')
        # HACK: remove the 'dtype: object' part of Series's string
        # representation.
        nb_results = str(nb_results).replace('\ndtype: object', '')
        print(f'\nNaive Bayes:\n\n{nb_results}\n')
        
        # Compute and print character probs with the NN model.
        X = models.get_nn_X([tokens], word_to_id)
        nn_pred_probs = nn_model.predict(X)[0]
        nn_results = pd.Series(nn_pred_probs, index=class_labels)
        nn_results = nn_results.map(lambda x: f'{x:.3%}')
        nn_results = str(nn_results).replace('\ndtype: object', '')
        print(f'Neural network:\n\n{nn_results}\n')


@contextmanager
def suppress_stderr():
    """Suppress anything written to stderr within the context manager. """
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        # Errors occuring while the context manager is open
        # will get raised here via `.throw(e)`. We don't handle them
        # but need the finally block to make sure stderr is reconnected.
        try:
            yield
        finally:
            sys.stderr = old_stderr

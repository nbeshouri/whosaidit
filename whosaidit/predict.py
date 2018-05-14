"""
This module contains functions that score articles using a given model.

Todo:
    * Add support for other models.

"""

import os
from . import utils, data, scraper
from keras.models import load_model
from . import transforms

data_dir_path = os.path.join(os.path.dirname(__file__), 'data')

def predict_probs_nn(model, text):
    normalized = transforms.normalize(text)
    tokens = transforms.to_tokens(text)
    
    

def predict_probs_nb(model, text):
    pass




def predict_score(article_name):
    """
    Predict the score of an article.
    
    Args:
        article_name (str): The cannonical string name of an article.
            E.g., ``Wikipedia`` or ``World_Health_Organization``. It
            should also work without the underscores.
    
    Returns:
        float: The predicted quality of the model on a 0 to 5 scale.

    """
    model = load_model(os.path.join(data_dir_path, 'nn_model.hdf5'),
                       custom_objects={'r2': utils.r2})
    scraped_data = scraper.scrape_article(article_name)
    data_df = data.import_raw_data(article_data=[scraped_data])
    transformed_df = data.transform_data(data_df)
    X = transformed_df[data.simple_features]
    return model.predict(X.as_matrix())[0, 0]

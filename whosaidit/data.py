#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import joblib
from .scraper import scrape_data_file_path
from sklearn.model_selection import train_test_split

# These are the scraped field values that can be converted to floats or ints.
numeric_columns = [
    'ID', 'Page size', 'Total edits', 'Editors', 'Bugs', 
    'Bugs', 'Page watchers', 'Pageviews (60 days)', 'IP edits', 
    'Bot edits', '(Semi-)automated edits', 'Reverted edits', 
    'Max. text added', 'Max. text deleted',
    'Average time between edits (days)', 'Average edits per user', 
    'Average edits per day', 'Average edits per month', 'Average edits per year',
    'Edits in the past 24 hours', 'Edits in the past 7 days', 
    'Edits in the past 30 days', 'Edits in the past 365 days', 
    'Edits made by the top 10% of editors', 'Links to this page', 'Redirects',
    'Links from this page', 'External links', 'Categories', 'Files', 'Templates',
    'Characters', 'Words', 'Sections', 'References', 'Unique references', 
    'Accounts', 'IPs', 'Major edits', 'Minor edits', 'Top 10%', 'Bottom 90%'
]

# This complex feature set wasn't used in the actual model. It's potentially
# more powerful, but it seemed to make the models more temperamental and I didn't 
# have time to debug it.
complex_features = [
    'Account edits (ratio)', 'Age (days)', 'Average edits per day (log)', 
    'Average edits per user', 'Bot edits (ratio)', 'Bugs', 'Categories', 
    'Words (log)', 'Editors (log)', 'External links (log)', 'File links (log)', 
    'Total edits (log)', 'Page watchers (log)', 'Pageviews (60 days) (log)', 
    'Redirects', 'References', 'Reverted edits (ratio)', 'Sections', 
    'Semi-automated edits (ratio)', 'Unique references', 'Links to this page (log)', 
    'Links from this page'
]

# These are the features that were used in all presented models.
simple_features = [
    'Account edits (ratio)', 'Age (days) (log)', 'Average edits per day (log)', 'Bugs', 
    'Categories', 'Words (log)', 'Editors (log)', 'File links (log)', 'Redirects',
    'References', 'Reverted edits (ratio)', 'Sections', 'Semi-automated edits (ratio)', 
    'Unique references', 'External links (log)'
]


def import_raw_data(article_data=None):
    """Convert scraped key/value pairs into a `DataFrame`."""
    if article_data is None:
        article_data = joblib.load(scrape_data_file_path)
    
    df = pd.DataFrame(article_data)
    
    date_columns = ['First edit', 'Latest edit']

    def convert_col(col):
        col = col.str.extract(r'([\d.,]*)', expand=False)
        col = col.str.replace(',', '')
        col = col.fillna(0)
        return pd.to_numeric(col)

    df.dropna(subset=['Assessment'], inplace=True)

    # Drop the duplicates.
    df.drop_duplicates(subset=['ID'], inplace=True)

    # Filter out the other assessment types. Yes, somehow they're there.
    mask = df['Assessment'].str.match(r'(?:FA|A|GA|B|C|Start|Stub)$')
    df = df[mask]

    # Convert data types.
    df[numeric_columns] = df[numeric_columns].apply(convert_col)
    df[date_columns] = df[date_columns].apply(pd.to_datetime)

    # Drop the IPs column if IP edits is already present. It's the same
    # data but different versions of the scrapper handled this differently.
    if 'IP edits' in df.columns:
        df.drop('IPs', axis=1, inplace=True)

    # Give some columns more descriptive names.
    mapper = {'(Semi-)automated edits': 'Semi-automated edits', 
              'Top 10%': 'Edits by top 10%', 'Accounts': 'Account edits',
              'Files': 'File links', 'IPs': 'IP edits'}
    
    df.rename(mapper, axis=1, inplace=True)
    df.sort_index(inplace=True, axis=1)
    
    return df


def transform_data(df):
    """Transform some of the features."""
    df = df.copy()
    
    # Add some columns.
    df['Age (days)'] = (pd.datetime.now() - df['First edit']).dt.days
    asses_map = {'Stub': 0, 'Start': 1.0, 'C': 2.0, 'B': 3.0, 'A': 4.0, 'GA': 4.0, 'FA': 5.0}
    df['Rating'] = df['Assessment'].map(asses_map)
    
    # Create ratio versions of some columns.
    edit_cols = [
        'Edits by top 10%', 'IP edits', 'Bot edits', 'Semi-automated edits', 
        'Account edits', 'Minor edits', 'Major edits', 'Reverted edits'
    ]
    for col_name in edit_cols: 
        df[col_name + ' (ratio)'] = df[col_name] / df['Total edits']

    # Create log scaled versions of some columns.
    cols_to_log = [
        'Average edits per day', 'Average edits per user', 'Editors', 'External links',
        'File links', 'Page watchers', 'Pageviews (60 days)', 'Words', 'Total edits', 
        'Age (days)', 'Major edits', 'Account edits', 'Bot edits', 'Links to this page'
    ]
    new_col_names = [col_name + ' (log)' for col_name in cols_to_log]    
    df[new_col_names] = df[cols_to_log].applymap(lambda x: np.log(x) if x > 0 else 0)    
 
    return df


def get_X_y(features, df):
    """Return the test/train split for the given features."""
    X = df[features]
    y = df['Rating']
    return train_test_split(X, y, random_state=42, test_size=0.2)

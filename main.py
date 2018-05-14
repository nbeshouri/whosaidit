#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module is a stub that allows you to run the command line interface
directly without installing the package. (Trying to run command_line.py
directly would break its relative imports.)

"""

import os
import joblib
from whosaidit.transforms import extract_text_features
import whosaidit.scraper
import whosaidit.models
import whosaidit.utils

if __name__ == '__main__':
    whosaidit.models.test()
    
    # data_dir_path = os.path.join(os.path.dirname(__file__), 'whosaidit', 'data')
    # path = os.path.join(data_dir_path, 'buffy_dialogue_elements.pickle')
    # data = joblib.load(path)
    # data = extract_text_features(data)
    # file_name = 'buffy_dialogue_elements_with_text_features.pickle'
    # whosaidit.utils.archive_data(file_name)
    # path = os.path.join(data_dir_path, file_name)
    # joblib.dump(data, path)

    

    # whosaidit.scraper.BuffyScraper().scrape()


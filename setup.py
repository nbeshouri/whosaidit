#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='whosaidit',
      version='0.1',
      description='TODO',
      python_requires='>=3.6',
      author='Nicholas Beshouri',
      author_email='nbeshouri@gmail.com',
      packages=['whosaidit'],
      package_data={'whosaidit': ['data/*.pickle', 'data/*.hdf5']},
      include_package_data=True,
      zip_safe=False, 
      install_requires=['pandas', 'numpy', 'bs4', 'sklearn', 'keras', 
                        'click', 'joblib', 'textblob', 'spacy', 'munch'],
      entry_points={'console_scripts': ['whosaidit=whosaidit.command_line:cli']})

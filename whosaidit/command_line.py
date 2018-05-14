"""
This module contains the the tool's command line interface.

"""

import os
import sys
from contextlib import contextmanager
import click


@click.group()
def cli():
    """Dummy function used by `Click`."""
    pass


@click.command()
@click.argument('article_name')
def predict(article_name):
    """Predict the rating of the given article and print result."""
    # This is a hack to prevent keras and tensor flow from dumping
    # confusing messages to the command line when imported.
    with suppress_stderr():
        import keras
    from .predict import predict_score
    rating = predict_score(article_name)
    print(f'The model gives the article "{article_name}" a rating of {rating:.2f}.')


@click.command()
@click.option('-n', type=int)
@click.option('-f', type=int, default=100)
@click.option('--save', type=bool, default=True)
def scrape(n, f, save):
    from .scraper import scrape_articles
    scrape_articles(article_count=n, log_freq=f, save=save)


cli.add_command(predict)
cli.add_command(scrape)


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

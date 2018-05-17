"""
This module contains text transformation functions. It should probably
also contain some of the data manipulation funtions in models.py.

"""

import re
import spacy


nlp = spacy.load('en_core_web_md')
# Due to a bug, this language model doesn't contain
# stop words, so we fix that here.
for word in nlp.Defaults.stop_words:
    lex = nlp.vocab[word]
    lex.is_stop = True


def normalize(text):
    """Make text lowercase and make some replacements."""
    text = text.lower()
    
    replacements = [
        (r'\[.*\]', ''), # Remove meta-text annotation.
        (r'\(.*\)', ''),  
        (r'[——]', ' '),
        (r'--', ' '),
        (r'-\s', ' '),
        (r'doo+m', 'doom'),
        (r'\?+', '?'),
        (r'!+', '!'),
        (r'\.+', '.'),
        (r'aw+', 'aw'),
        (r'hm+', 'hm'),
        (r'no+', 'no'),
        (r'a+h+', 'ah'),
        (r'soo+n', 'soon'),
        (r'[“”]', '"'),
        (r"’", r"'")
    ]
    
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    
    return text


def to_lemmas(text):
    """Convert words in `text` to a string of lemmas."""
    lemmas = []
    for token in nlp(text):
        if not token.is_punct and not token.is_stop and len(token.text) < 15:
            if token.lemma_ != ' ':
                lemmas.append(token.lemma_.strip())
    return ' '.join(lemmas)


def to_tokens(text):
    """Convert words in `text` to a string of tokens."""
    tokens = []
    allowed_punct = set('.?!,')
    for token in nlp(text):
        if ((token.is_punct and token.text not in allowed_punct)
                or len(token.text) > 15
                or token.text == ' '):
            continue
        tokens.append(token.text)
    return ' '.join(tokens)


def get_polarity(text):
    """Get average pos/neg polarity of a string."""
    from textblob import TextBlob
    blob = TextBlob(text)
    return blob.sentiment.polarity
    

def extract_text_features(episode_munches):
    """Add lemma, token, and polarity to episode Munch objects."""
    output = []
    for episode in episode_munches:
        episode = episode.copy()
        normalized = normalize(episode.text)
        episode.lemmas = to_lemmas(normalized)
        episode.tokens = to_tokens(normalized)
        episode.polarity = get_polarity(episode.text)
        output.append(episode)
    return output

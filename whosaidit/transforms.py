from textblob import TextBlob
import re
import spacy

# TODO: 23889  S10E12  Stench and Stenchibility  pfffffff  locker tonya keep tap shoe right dan...       115 -0.047321  Bender  Pfffffff!\nSee that locker? Tonya keeps her ta...  pfffffff ! \n see that locker ? tonya keeps he...


nlp = spacy.load('en_core_web_md')
for word in nlp.Defaults.stop_words:
    lex = nlp.vocab[word]
    lex.is_stop = True


def normalize(text):
    text = text.lower()
    
    replacements = [
        (r'\[.*\]', ''), # Remove meta-text annotation.
        (r'\(.*\)', ''),  
        (r'[——]', ' '),  # Dashes to spaces.
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
    lemmas = []
    for token in nlp(text):
        if not token.is_punct and not token.is_stop and len(token.text) < 15:
            if token.lemma_ != ' ':
                lemmas.append(token.lemma_.strip())
    return ' '.join(lemmas)


def to_tokens(text):
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
    blob = TextBlob(text)
    return blob.sentiment.polarity
    

def extract_text_features(episodes):
    output = []
    for episode in episodes:
        episode = episode.copy()
        normalized = normalize(episode.text)
        episode.lemmas = to_lemmas(normalized)
        episode.tokens = to_tokens(normalized)
        episode.polarity = get_polarity(episode.text)
        output.append(episode)
    return output

# whosaidit

A not that great dialogue attributer.

## Getting Started

### Prerequisites

* Python 3.6 or higher
* `numpy`
* `pandas`
* `bs4`
* `sklearn`
* `keras`
* `click`
* `joblib`
* `spacy`
* `textblob`
* `munch`

### Installation

The tool can be installed using `pip` or `setuptools`, but you can also run it directly from the repo folder using `main.py` as long as you have all the prerequisites.

```bash
cd <repo directory>
pip install .
```

Or:

```bash
python setup.py install
```

### Usage

If you installed it:

```bash
whosaidit
```

If you didn't:
```bash
python main.py
```
## Known issues

1. You need to have `spacy`'s 'en_core_web_md' language model installed or errors will likely occur.
2. `setup.py` is configured but is untested. You should probably run it with `main.py`.

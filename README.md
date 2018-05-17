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

1. GitHub won't host the data so you'll have to download the data folder from [here](https://www.dropbox.com/sh/qr1y9lysawj1ukq/AACTLV2J9xW2TUASREZC3IPQa?dl=0) and manually place it inside the `whosaidit` folder.
2. You need to have `spacy`'s 'en_core_web_md' language model installed or errors will likely occur.
3. `setup.py` is configured but is untested. You should probably run it with `main.py`.
4. The embedding files are not included and there won't be a nice warning if you try to use them.
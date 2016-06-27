"""
Downloads the necessary NLTK corpora.
Usage: ::
    $ python -m questions.download_corpora
"""
if __name__ == '__main__':
    from nltk import download

    CORPORA = ['punkt']

    for corpus in CORPORA:
        download(corpus)

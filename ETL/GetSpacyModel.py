import spacy
import subprocess
import sys
import urllib.request
import zipfile
import os
import argparse

UNCASED = "https://nlp.stanford.edu/data/glove.42B.300d.zip"
# CASED = "https://nlp.stanford.edu/data/glove.840B.300d.zip"

DOWNLOAD = "download"
SAVED_ZIP = os.path.join(DOWNLOAD, "glove.model.zip")
GLOVE_TXT = os.path.join(DOWNLOAD, "glove.42B.300d.txt")

LOCATION = '.'

#step by step download and build Spacy model
def getModels(location='.', verbose=True):
    if verbose:
        print("Downloading...")

    #Download vector database
    urllib.request.urlretrieve(UNCASED, SAVED_ZIP)

    if verbose:
        print("Unzipping")

    #unzip text file to use in build
    with zipfile.ZipFile(SAVED_ZIP, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(DOWNLOAD, '.'))

    if verbose:
        print("Creating Spacy Model")

    #create models
    #one from full Glove database
    #one a subset for testing purposes
    load_word_vectors(os.path.join(location, "glove_model"), GLOVE_TXT, verbose)
    load_word_vectors(os.path.join(location, "test_model" ), "glove.test.300d.txt", verbose)

    if verbose:
        print("Done")

#starts spacy subprocess to build a model with accociated vectors
def load_word_vectors(model_name, word_vectors, verbose=True):
    subprocess.run([sys.executable,
                    "-m",
                    "spacy",
                    "init",
                    "vectors",
                    "en",
                    word_vectors,
                    model_name
                ])

    if verbose:
        print (f"New spaCy model created with word vectors. File: {model_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Get Spacy Model',
                    description='Train on input data to produce 2 different models for usage in front end')

    parser.add_argument('-o', '--output', help=f"Diretory to save models in, default='{LOCATION}'", default=LOCATION)
    parser.add_argument('-v', '--verbose', help="Flag to add some verbosity to this process, default=False", default=False, action='store_true')

    args = parser.parse_args()

    getModels(location=args.output, verbose=args.verbose)

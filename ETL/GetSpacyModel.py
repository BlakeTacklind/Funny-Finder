import spacy
import subprocess
import sys
import urllib.request
import zipfile

UNCASED = "https://nlp.stanford.edu/data/glove.42B.300d.zip"
# CASED = "https://nlp.stanford.edu/data/glove.840B.300d.zip"

SAVED_ZIP = "download/glove.model.zip"
GLOVE_TXT = "download/glove.42B.300d.txt"

#step by step download and build Spacy model
def getModels(verbose=True):
	if verbose:
		print("Downloading...")
	
	#Download vector database
	urllib.request.urlretrieve(UNCASED, SAVED_ZIP)

	if verbose:
		print("Unzipping")
	
	#unzip text file to use in build
	with zipfile.ZipFile(SAVED_ZIP, 'r') as zip_ref:
	    zip_ref.extractall('download/.')

	if verbose:
		print("Creating Spacy Model")
	
	#create models
	#one from full Glove database
	#one a subset for testing purposes
	load_word_vectors("glove_model", GLOVE_TXT, verbose)
	load_word_vectors("test_model", "glove.test.300d.txt", verbose)

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
                        ]
                    )

    if verbose:
    	print (f"New spaCy model created with word vectors. File: {model_name}")

if __name__ == '__main__':
	getModels()
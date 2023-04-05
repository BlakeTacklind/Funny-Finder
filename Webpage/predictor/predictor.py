
import spacy
from spacy.attrs import ORTH, NORM

import numpy as np
import pandas as pd
import tensorflow as tf

import math

import random

import os
script_dir = os.path.dirname(__file__)

MODEL_1 = os.path.join(script_dir, "models", "model0")
MODEL_2 = os.path.join(script_dir, "models", "model1")
MODELS = [MODEL_1, MODEL_2]

MODEL_IDX = 0

SPACY_MODEL = os.path.join(script_dir, "glove_model")
#a spacy model for testing purposes
TEST_MODEL = os.path.join(script_dir, "test_model")

class Predictor:
	"""
	A class to handle all predictions of sentences into a predicted
	"""
	embeddings = None

	def __init__(self, embeddedFile = SPACY_MODEL, model=MODEL_IDX):
		print("loading tokenizer")
		self.loadTokenizer(embeddedFile)
		print("loading model")
		self.currentModel = None
		self.changeModel(MODEL_IDX)
		print("loading completed")


	def toEmbeddings(self, text):

		tokens = self.tokenizer(text.lower())

		return pd.DataFrame([{'word': val, 'embed':val.vector} for val in tokens])

	def getEmbed(self, value):
		embeddings = self.embeddings

		if str(value) in embeddings.keys():
			return embeddings[str(value)]
		else:
			return None

	def loadTokenizer(self, embeddedFile):
		# nlp = spacy.load("en_core_web_lg", exclude=["ner"])
		nlp = spacy.load(embeddedFile, exclude=["ner"])

		self.tokenizer = nlp.tokenizer

	def tokenExists(self, token):
		return self.tokenizer.vocab.has_vector(token)

	def predict(self, text):
		#get a dataframe which includes the embeddings and words
		embedded = self.toEmbeddings(text)

		#get data prepared to go into model
		prepared = self.toModelValues(embedded)

		#predict the values
		predictions = self.model.predict(prepared)

		#convert the text to an approprivate output array
		output = self.getOutput(predictions, embedded)

		return output

	def toModelValues(self, embeddings):
		#remove nulls because we don't train on them
		trimmed = self.dropEmpty(embeddings)

		#convert to a numpy array
		np_array = np.array([trimmed['embed'].to_list()])

		return np_array

	def loadModel(self, model):
		self.model = tf.keras.models.load_model(model)

	def changeModel(self, model_index):

		if self.currentModel == model_index:
			#model is already correct
			return False

		self.loadModel(MODELS[model_index])
		self.currentModel = model_index

		return True

	def getCurrentModel(self):
		return self.currentModel

	def getModelDescription(self):
		return [
	        "Model is currently 0",
	        "Model is currently 1",
        ][self.currentModel]

	#combines and generates an output
	def getOutput(self, predictions, embedded):
		trimmed = self.dropEmpty(embedded)

		#create a new dataframe that has the predictions
		pred_df = pd.DataFrame(
			index = trimmed.index,
			data = predictions[0,:len(trimmed),0],
			columns = ['prediction'],
		)

		#add the predicted values to the table
		completeTable = embedded.join(pred_df)

		#function to convert a single row into the appropriate output for the flask app
		def convertRow(row):
			idx, data = row
			return {
				'word': data['word'],
				'value': data['prediction'],
			}

		#get the output as a list with just the word and the prediction
		return list(map(
			convertRow,
			completeTable.iterrows(),
		))


	def dropEmpty(self, embeddings):
		return embeddings[embeddings.word.apply(lambda e: e.has_vector)]

	# return a random answers for testing purposes
	def randomPredict(self, text):
		options = [True, False, None]
		weights = [3, 5, 1]

		tokens = self.tokenizer(text)

		return [{
			'word': word.text_with_ws,
			'prediction': random.choices(options, weights = weights)[0],
			'value': random.choices([random.random(), float("nan")], weights = [3,1])[0]
		} for word in tokens]

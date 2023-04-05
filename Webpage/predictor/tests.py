import unittest
import numpy as np
import math

from predictor import Predictor, TEST_MODEL

class TestPredictor(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.P = Predictor(TEST_MODEL)

	# Test if it can load embeddings
	def test_loading(self):
		embeddings = self.P.embeddings

		self.assertTrue(self.P.tokenExists("this"))
		self.assertFalse(self.P.tokenExists("missing"))

	def test_tokenize_good(self):
		TEST_LINE = "this is a test"

		output = self.P.toEmbeddings(TEST_LINE)

		self.assertEqual(len(output), 4)

		for idx, line in output.iterrows():
			self.assertEqual(line['word'].orth_, TEST_LINE.split()[idx])

			self.assertTrue(np.all(line['embed'] != 0))


	def test_tokenize_missing(self):
		TEST_LINE = "this is a test missing test"

		output = self.P.toEmbeddings(TEST_LINE)

		self.assertEqual(len(output), 6)

		for idx, line in output.iterrows():
			self.assertEqual(line['word'].orth_, TEST_LINE.split()[idx])

			if idx != 4:
				self.assertTrue(np.all(line['embed'] != 0))
			else:
				self.assertTrue(np.all(line['embed'] == 0))

	#TODO fix it so it doesn't use lower case words
	def test_tokenize_case_insensitive(self):
		TEST_LINE = "This is a Test"

		output = self.P.toEmbeddings(TEST_LINE)

		self.assertEqual(len(output), 4)

		for idx, line in output.iterrows():
			self.assertEqual(line['word'].orth_, TEST_LINE.lower().split()[idx])

			self.assertTrue(np.all(line['embed'] != 0))


	def test_predict_good(self):
		TEST_LINE = "this is a test"

		output = self.P.predict(TEST_LINE)

		self.assertEqual(len(output), 4)

		for idx, line in enumerate(output):
			self.assertEqual(line['word'].orth_, TEST_LINE.split()[idx])

			self.assertTrue(not math.isnan(line['value']))

	def test_predict_good(self):
		TEST_LINE = "this is a test missing test"

		output = self.P.predict(TEST_LINE)

		self.assertEqual(len(output), 6)

		for idx, line in enumerate(output):
			self.assertEqual(line['word'].orth_, TEST_LINE.split()[idx])

			if idx != 4:
				self.assertTrue(not math.isnan(line['value']))
			else:
				self.assertTrue(math.isnan(line['value']))

if __name__ == '__main__':
    unittest.main()

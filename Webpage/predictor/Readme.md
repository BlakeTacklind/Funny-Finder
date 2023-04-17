# Predictor

Python Module with a simple interface to use this model/predictor

## Usage

```python
from predictor import Predictor

#loads the models, takes some time
predictor = Predictor()

INPUT_TEXT = "Some string of arbitrary length of input text"
predictions = predictor.predict(INPUT_TEXT)
```

`predict` Returns a list of dictionaries. One item for each token. Each with `word` and `value`. Word is a lower case string. Value is between 0 and 1 or None in case the token was not found.

### Change the Model

There are 2 models that can be used

```python
#There are only 2 models at the moment
#model 0 and model 1. Defaults to model 0
NUMBER = 1
predictor.changeModel(NUMBER)
```

## Tests

This part has some unit tests that can be run with:

`python tests.py`

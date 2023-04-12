# Front End

A small Flask front end for using and displaying model results trained by [ETL Pipeline](../ETL).


## Usage

Enter text into the text box an then you can click on predict button.

The page will change and have the predictions on the input text and at the bottom you can change how you wish to look at the data. For instance you can change it to use a gradient of white to green background for how likely the token is in a punchline.

## Models

### Model 0

Trained against where multiple tokens right before the laughter are marked as the positive case

Poor numbers, likely due to the construction process of the punchline labels

**Loss:** 0.2421
**AUC:** 0.5921

### Model 1

Trained against token right before the laughter is the positive case, everything else is false

Leads to it mainly predicting on puctuation as that marks the end of a sentence

**Loss:** 0.1128
**AUC:** 0.7572

## Install

How to install and run this flask app

### Local

```bash
#likely want to use a virtual env
pip -m virtualenv flask_env
flask_env/Scrits/activate

#download and extract the vocabulary model
#https://storage.googleapis.com/funny_finder_models/glove_model.tar.gz
#to ./predictor/glove_model

#install requirements
pip install -r requirements.txt

#run the script
python -m flask run
```

Then visit the site [127.0.0.1:5000](http://127.0.0.1:5000)

### Docker

If using Docker

```bash
#build docker image
docker build . -t laugh_website/server:latest

#run docker image
docker run -d -p 5000:5000 laugh_website/server:latest
```

visit the site [http://localhost:5000/](http://localhost:5000/)

## TODO

Could use a simple static front end and have the application just be the backend. Leads to a web page that loads instantly and a much more expandable backend at the cost of a slightly more complex architecture.


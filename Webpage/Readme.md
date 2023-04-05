# Front End

A small Flask front end for using and displaying model results trained by [ETL Pipeline](../ETL).


## Usage

First Page

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
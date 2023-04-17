# Extract Tranform Load

For simplicity this process is done all in local space. This folder is about transforming the selected data from [Data Selector](../Data%20Selector/) and turning it into a model usable by [Webpage](../Webpage/).

## Script Description

The script order, should be turned into a Spark (or similar) pipeline. All scripts have some command line options. Use `-h` for help.

### [Audio Grabber](AudioGrabber.py)

**Intake:** videos.csv (a csv produced by Data Selector)

Connects to YouTube Api and downloads the audio for every useful video storing it in the appropriate data format in audio/ folder

**Output:** audio/\*.mp3

### [Produce Laugh Data](ProduceLaughData.py)

**Intake:** audio/\*.mp3 (produced by Audio Grabber)

Uses a [Laughter Detector](https://github.com/jrgillick/laughter-detection) to check audio for segments of laughter. Produces a json with the laughter probability numbers over time. When used with a lowpass filter these numbers produce good boolean predictions of laughter segments

**Output:** laughs.json

### [Transcript Grabber](TranscriptGrabber.py)

**Intake:** videos.csv (a csv produced by Data Selector)

Simply downloads the transcripts from YouTube and stores them as a singel json. This data has timing by sentence

**Output:** transcript.json

### [Transcript To Word](TranscriptToWord.py)

**Intake:** transcript.json (from Transcript Grabber), audio/\*.mp3 (produced by Audio Grabber)

Transforms sentence level transcript data into word level transcriptions. Attempts to use forced alignment (audio and text synchronization) to get more accurate word level timing.

**Output:** word_transcript.csv

### [Load Vector Model](GetSpacyModel.py)

Downloads [GloVe embedding](https://nlp.stanford.edu/projects/glove/) and creates a spacy model from it. It is more complete then the Spacy model

**Output:** glove_model/; test_model/ (test model is for unit tests) 

### [Combine Data](TableMaker.py)

**Intake:** word_transcript.csv, laughs.json, GloVe embedding

Combines data into a single table with 2 possible target values

**Output:** trainable.csv

### [Train the Model](Train.py)

**Intake:** trainable.csv

Train based on word embeddings and one of the 2 possible target values. Creates an LSTM and Dense Neural Net that can be used by [Webpage](../Webpage)

**Output:** model

## TODO

Make a execution pipeline with Spark or similar workflow

# Funny Finder

An end to end NLP project to figure out where the punchlines are in text

## Corpus

Data from [YouTube Standup](https://www.youtube.com/hashtag/standup) routines are used for training and testing data. See [Data Selector](./Data%20Selector/) for selection and its criteria

## [ETL](./ETL/)

There is a pipeline for the extraction and transformation of data from YouTube to a trained model

## [Prediction Website](./Webpage/)

A small Flask app set up to use models that are created by the ETL pipeline and predict on user supplied text

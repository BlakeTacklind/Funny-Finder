#!/usr/bin/env python

import spacy
from spacy.attrs import ORTH, NORM
# import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from pathlib import Path
import re
from laughter_detection.laugh_segmenter import get_laughter_instances
import argparse

SPACY_MODEL = "glove_model"
LAUGH_DATA = 'laughs.json'
WORD_DATA = 'word_transcript.csv'
OUTPUT_FILE = "trainable.csv"

def makeTable(wordTranscript, laughData, verbose=True):

    if verbose:
        print("Loading Embedding")

    fullEmbed = getFullEmbedding(wordTranscript)
    laughValuesDict = getLaughData(laughData)

    print("Getting Word delimiters")
    punchWords = [getPunchWord(data, fullEmbed) for data in laughValuesDict if data['id'] in fullEmbed.index]


    #build data for tokens before laughter
    df = fullEmbed.join(pd.concat(punchWords))

    print("Removing Bad data")
    #clean up by removing videos with low word counts or don't contain laughs
    #this fiters out videos that often don't have an audience or that
    #the transcript is in the video (and not the closed captions)
    laughless = findLaughless(df)
    lowWord = findLowWord(laughValuesDict, fullEmbed)
    df = removeBadVideos(df, laughless + lowWord)


    print("Finding Punchlines")
    #build data for group of words right before the laughter
    df['cata2'] = df.groupby('id', group_keys=False).apply(punchline)

    #clean up data by removing tokens that don't exist
    df = removeUnknownTokens(df)

    return df

# Build Buinary Data

MS_TO_SEC = 1000
def getPunchWord(laughDict, embeddings):
    idx = laughDict['id']

    #convert seconds to milliseconds
    laugh_table = getLaughTable(laughDict['values']) * MS_TO_SEC

    #check for if laughter is between each word
    output = embeddings.loc[idx].start.\
        rolling(window=2, closed='right').\
        apply(lambda val: checkInside(val, laugh_table)).\
        rename("cata")

    output = pd.concat({idx: output}, names=['id'])

    #shift for alignment
    output = output.shift(-1).astype(bool)

    #special case of the last token
    #use the token's time and the end of the video's time
    s = pd.Series([embeddings.loc[idx].iloc[-1].start, laughDict['length'] * MS_TO_SEC])
    output.iloc[-1] = checkInside(s, laugh_table)

    return output

#Get list of Start and
def getLaughTable(values):
    return pd.DataFrame({'start': start, 'stop': stop} for start, stop in get_laughter_instances(values))

#find elements of series that are between the two values provided
def checkInside(vals, series):
    return np.any((series > vals.iloc[0]) & (series < vals.iloc[1]))

#Time before laughter with is included in punchline
PUNCH_DIFF = 100

#get the punchline from videos
#TODO should likely label some videos manually and create a simple classifier
#This could use all the laughter and potentially video data to classify punchlines
def punchline(video):
    #finds all words at or before the laughter by a specific time margin
    return video[video.cata].start.apply(
        lambda punchTime: (video.start > punchTime - PUNCH_DIFF) & (video.start <= punchTime)
    ).any()

### Get embeddings
# Section for 

def getFullEmbedding(word_file):
    word_df = pd.read_csv(word_file, index_col=['id', 'number'])
    return getFullEmbeddingsTable(word_df)


def getFullEmbeddingsTable(df):
    embeddings = getWordEmbeddings(df)

    #drop any null values, should be none
    partialDf = pd.DataFrame(embeddings.tolist())
    dropped_df = partialDf.embed.dropna()

    #add prefix to embedded columns
    table = pd.DataFrame(
        dropped_df.tolist(),
        index=dropped_df.index
    ).add_prefix("embed")

    full = partialDf[['start', 'token']].join(table).set_index(
        pd.MultiIndex.from_tuples(partialDf['index'], names=["id", "number"]),
        drop=True
    )

    #Some words get split into multiple tokens
    #reflect that by reseting the index
    full = full.groupby('id').apply(lambda x: x.reset_index(drop=True))

    full.index.rename("number", level=1, inplace=True)

    return full

def getWordEmbeddings(df):
    nlp = spacy.load(SPACY_MODEL)
    tokenizer = nlp.tokenizer
    return np.concatenate([toWordEmbed(current, tokenizer) for current in df.iterrows()])

def toWordEmbed(rowItr, tokenizer):
    idx, row = rowItr

    tokens = tokenizer(row.word)
    
    return [{'index':idx, 'start': row.start, 'token': t, 'embed': t.vector} for t in tokens]

### Load Laughter Data
def getLaughData(laugh_json):
    with open(laugh_json, 'r') as laughFile:
        return json.loads(laughFile.read())

### Section on cleanup of the data

#TODO use some Binary Classification to distinguish these
def removeBadVideos(df, badIds):
    return df[~df.index.get_level_values('id').isin(badIds)]

#Find videos with an abnormally low word count
def findLowWord(laughDict, embeddings):

    laughDf = pd.DataFrame(laughDict).set_index('id', drop=True)

    wordsPerSecond = embeddings['token'].groupby('id').count() / laughDf.length

    #less then 1 word per second is likely a bad video
    return wordsPerSecond[(wordsPerSecond < 1)].index.tolist()

#return a list of videos that contain no detected laughter
def findLaughless(df):
    laughful = df.cata.groupby('id').apply(np.any)
    return list(laughful[~laughful].index)

#filter out rows that don't have a vector for the word
def removeUnknownTokens(df):
    return df[df.token.apply(lambda x: x.has_vector)]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='Table Maker',
                    description='Makes a single table from all data sources')

    parser.add_argument('-w', '--word', help=f"Input single word CSV likely from TranscriptToWord, default={WORD_DATA}", default=WORD_DATA)
    parser.add_argument('-l', '--laugh', help=f"Input json likely from ProduceLaughData, default={LAUGH_DATA}", default=LAUGH_DATA)
    parser.add_argument('-o', '--output', help=f"CSV to output combined table as, default={OUTPUT_FILE}", default=OUTPUT_FILE)

    args = parser.parse_args()

    out = makeTable(Path(args.word), Path(args.laugh))

    print("Saving Table")

    out.to_csv(args.output)

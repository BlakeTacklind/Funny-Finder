#!/usr/bin/env python

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import TranscriptsDisabled
import csv
import json
import argparse

DataFile = '../Data Selector/videos.csv'

transcriptFile = 'transcript.json'

def getGoodData(inputFile):

    data = []
    #
    with open(inputFile, 'r') as f:
        data = list(csv.DictReader(f))

    return [{'id': item['id'], 'lang':item['lang']} for item in data if item['useable'] == "True"]

def getTranscripts(inputFile):
    data = getGoodData(DataFile)

    trans = []
    for row in data:
        print(row['id'])
        try:
            trans.append({'tran': YouTubeTranscriptApi.get_transcript(row['id'], languages=[row['lang']]), 'id':row['id']})
        except TranscriptsDisabled:
            print(f'no transcript for {row["id"]}')

    return trans

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Transcript Grabber',
                    description='Fetches Transcripts from YouTube based on input csv')

    parser.add_argument('-i', '--input', help=f"Input CSV likely from Data Selector, default={DataFile}", default=DataFile)
    parser.add_argument('-o', '--output', help=f"Json to output data to, default={transcriptFile}", default=transcriptFile)

    args = parser.parse_args()

    trans = getTranscripts(parser.input)

    with open(parser.output, 'w') as f:
        f.write(json.dumps(trans))

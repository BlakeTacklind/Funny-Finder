#!/usr/bin/env python

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import TranscriptsDisabled
import csv
import json

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
	trans = getTranscripts(DataFile)

	with open(transcriptFile, 'w') as f:
		f.write(json.dumps(trans))

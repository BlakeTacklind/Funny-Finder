#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals
import yt_dlp
from pathlib import Path
import csv
import argparse

DataFile = '../Data Selector/videos.csv'
AUDIO_LOCATION = 'audio'
YOUTUBE_PREAMBLE = 'http://www.youtube.com/watch?v='

class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


# def progressHook(d):
#     if d['status'] == 'finished':
#         print('Done downloading, now converting ...')

def getGoodData(inputFile):

    data = []
    #
    with open(inputFile, 'r') as f:
        data = list(csv.DictReader(f))

    return [{'id': item['id'], 'lang':item['lang']} for item in data if item['useable'] == "True"]

def audioGrab(inputFile, audioLocation, verbose=True):

    data = getGoodData(inputFile);

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{audioLocation}/%(id)s.%(ext)s',
        'logger': MyLogger(),
        #no need to check download progress
        'progress_hooks': [],
    }

    def GetVideo(id):
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"{YOUTUBE_PREAMBLE}{id}"])

    #Get already finished
    completed = [x.name.split('.')[0] for x in Path(audioLocation).iterdir()]

    incomplete = [item['id'] for item in data if item['id'] not in completed]

    if verbose:
        print(f"Downloading {len(incomplete)} files:")

    for idx, ele in enumerate(incomplete):
        try:
            GetVideo(ele)
            if verbose:
                print(f"\rFinished file {idx + 1} / {len(incomplete)}", end="")
        except Exception as e:
            print(ele)
            print(e)

    if verbose:
        print("\nDone")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Audio Grabber',
                    description='Fetches Audio from YouTube based on input csv')

    parser.add_argument('-i', '--input', help=f"Input CSV likely from Data Selector, default={DataFile}", default=DataFile)
    parser.add_argument('-o', '--output', help=f"Diretory to download audio files to, default={AUDIO_LOCATION}", default=AUDIO_LOCATION)

    args = parser.parse_args()

    audioGrab(args.input, args.output)


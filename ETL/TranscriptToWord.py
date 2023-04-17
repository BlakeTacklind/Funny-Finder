#!/usr/bin/env python

import pandas as pd
import numpy as np
import json
import re
import pocketsphinx
from pathlib import Path
import os
import argparse

from pocketsphinx import Decoder, Config
from pydub import AudioSegment
from pydub.utils import mediainfo


TRANSCRIPTS = 'transcript.json'
AUDIO_PATH = "audio"
WORD_TRANSCRIPT = "word_transcript.csv"

#Constants
SAMPLE_RATE = 44100
#storage_in_bytes/(num_channels * sample_rate * (bit_depth/8))
BYTES_PER_SECOND = float(1 * SAMPLE_RATE *  (16/8))
PAD_TIME = 2

#Build a Table from input transcript and audio
def buildTable(transcriptData, audioPath):
    decoder = Decoder(samprate=SAMPLE_RATE, loglevel="FATAL")

    allThings = [toDict(table, decoder, audioPath) for table in transcriptData]

    allTable = pd.concat(
            (table['words'].set_index(pd.MultiIndex.from_product([[table['id']], table['words'].index], names=['id', 'word']))
            for table in allThings if type (table['words']) != list)
        )

    return allTable

#convert word transcript into a dict that is convertable to a pandas table
def toDict(data_item, decoder, audioPath):

    print(f"{data_item['id']}")
    trans, audio = getWordTranscript(data_item, decoder, audioPath)

    return {'id':data_item['id'], 'words':trans}


def getWordTranscript(data_item, decoder, audioPath):

    #construct string to audio file
    audioFile = os.path.join(audioPath, f"{data_item['id']}.mp3")

    #check for audio file existance
    if not Path(audioFile).is_file():
        print(f"no {audioFile}")

        return [], None

    #load the audio file
    audio = AudioSegment.from_file(audioFile, format="mp3")

    return getWordTable(audio, data_item['tran'], decoder), audio


def getWordTable(audio, transcript, decoder):

    def getPair(transcripts):
        for idx in range(len(transcripts) - 1):
            yield transcripts[idx], transcripts[idx + 1]

        yield transcripts[-1], None

    #get a list of tables, every table is a sentence,
    #every line is a word with its start time
    allTables, dataIndicator = zip(*[getPart(audio, current, decoder, nextTran)
            for current, nextTran in getPair(transcript)])

    #get some information about the amount of audio files decoded
    amount = len([x for x in dataIndicator if x == -1])
    if amount > 0:
        print(f"Couldn't get more accurate timing in {amount} of {len([x for x in dataIndicator if x != 0])} audio segments")

    #combine all the table into a single table
    combTable = pd.concat(allTables)

    #reset index to accept new word orders
    return combTable.reset_index(drop=True)

def getPart(audio, transcript, decoder, next_transcript = None):
    trimmed_transcript = cleanline(transcript['text'])

    start_time = transcript['start']
    duration_time = transcript['duration'] + 2 * PAD_TIME

    #check if we just trimmed away all the audio
    if trimmed_transcript == "":
        return pd.DataFrame(columns=['word', 'start']), 0

    #get audio segment in bytes
    audio_bytes = getRawAudioFromID(audio, getEarlyStartTime(start_time), duration_time, next_transcript)

    if len(audio_bytes) < 1:
        return pd.DataFrame(columns=['word', 'start']), 0

    #run forced alignment on audio segments
    timings = getWordTiming(trimmed_transcript, audio_bytes, start_time, decoder)

    return timings

def getWordTiming(text, audio, transcript_start, decoder):

    text_split = pd.DataFrame(word_punctuation.findall(text), columns=["word"])

    dataIndicator = 0

    if text_split.empty:
        return text_split, dataIndicator

    #check that word exists
    text_split2 = [word for word in text_split.word if decoder.lookup_word(word.lower()) is not None]

    text2 = " ".join(text_split2)

    if len(text_split2) == 0:
        text_split['start'] = transcript_start * 1000

        return text_split, dataIndicator

    decoder.set_align_text(text2)
    decoder.start_utt()
    decoder.process_raw(audio, full_utt=True)
    decoder.end_utt()

    itr = decoder.seg()

    if itr is None:
        # print(f"Could not handle the sentance: \"{text}\"")
        output = pd.DataFrame(columns=['word', 'start'])
        output.word = text_split

        #just assume all words start at the begining, for simplicity
        ratio = 0

        dataIndicator = -1
    else:
        decoded = pd.DataFrame({"word_d": d.word, "start":d.start_frame} for d in itr)

        #get last timestamp and duration of clip to get the ratio of time units
        #storage_in_bytes/(num_channels * sample_rate * (bit_depth/8))
        duration = len(audio) / BYTES_PER_SECOND

        endpoint = decoded.iloc[-1].start
        ratio = duration/endpoint

        decoded = decoded[decoded.apply(lambda x: not brackets_re.match(x.word_d), axis=1)]
        decoded['word'] = decoded.word_d.apply(lambda x: not_parathesis.search(x).group())
        decoded = decoded.reset_index(drop=True)

        #Merge the decoded audio and the original text into a single table
        output = fancyMerge(decoded, text_split)

        output = output.drop("word_d", axis=1)

        #since we pad the audio move the start time of the audio back by the Padded amount
        transcript_start = getEarlyStartTime(transcript_start)

        dataIndicator = 1


    #sometimes can't decode all the words in the text
    #fill in with pervious values
    output.start = output.start.fillna(method="ffill")
    #in the case where there is a single item without a time
    output.start = output.start.fillna(value=0)
    output = output.fillna(method='ffill', axis=1)

    #get the actual start time in milliseconds
    output.start = ((output.start * ratio) + transcript_start) * 1000
    #convert float to int
    output.start = output.start.astype('int32')

    return output, dataIndicator

def getRawAudioFromID(audio, start, duration, next_start = None):

    start_ms = start * 1000

    stop_time = start_ms + (duration * 1000)
    #videos occasionally have cross talk and have 2 simultanious transcripts
    #skip those
    if next_start and next_start['start'] > start:
        stop_time = next_start['start'] * 1000


    part = audio[start_ms: stop_time]

    tempFile = "audio.tmp"

    #uses ffmpeg under the hood so it has to save it to a file...
    #there might be a way to use temporary files but *shrug*
    part.export(tempFile, format="s16le")

    with open(tempFile, "rb") as f:
        audio_bytes = f.read()

    return audio_bytes

#Reuseable Regular Expresions
bm = re.compile(r"\[[^\]]+\] *")
bm2 = re.compile(r"\([^\)]+\) *")
frontdash = re.compile(r"(\s)\-+(\w)")
backdash = re.compile(r"(\w)\-+(\s)")
frontdash2 = re.compile(r"^\-+(\w)")
backdash2 = re.compile(r"(\w)\-+$")
brackets_re = re.compile(r"<[^>]+>")

def cleanline(line):
    out = line
    out = out.lower()
    out = out.replace('[\xa0__\xa0]', '*****')
    out = re.sub(bm, '', out)
    out = re.sub(bm2, '', out)
    out = out.replace('\n', ' ')
    out = out.replace('♪', '')
    out = out.replace('"', '')
    out = out.replace('’', "'")
    out = out.replace('’', "'")
    out = re.sub(frontdash, r'\1\2', out)
    out = re.sub(backdash, r'\1\2', out)
    out = re.sub(frontdash2, r'\1', out)
    out = re.sub(backdash2, r'\1', out)
    # out = out.replace('fu**', "fuck")
    return out


"""
interlace each row of the dataframe with NaN values
"""
def getInterlaced(dfOld):
    df = dfOld.copy()
    newDf = pd.DataFrame(None, index=pd.RangeIndex(len(df)*2 + 1))
    df['key'] = ((df.index + 1) * 2 - 1).astype('int32')
    return df.merge(newDf, how="right", left_on="key", right_index=True)


"""
Takes in 2 dataframes each with a column "word"
Returns a single dataframe that matches up word column
Both origin dataframes will be in order and NaNs will be filled in
for unmatched values.
Ambiguity will defer to the left dataframe

This is way over-engineered
"""
def fancyMerge(dfl, dfr):
    #interlace the initial dataframes with NaN values and cross them to get all
    #possible combinations
    cross = getInterlaced(dfl).merge(getInterlaced(dfr), how='cross')

    #remove combinations that are not valid (non-matching words or just NaN values)
    both_na = ~(cross.word_x.isna() & cross.word_y.isna())
    eq_or_na = cross.word_y.isna() | cross.word_x.isna() | (cross.word_y == cross.word_x)
    cross = cross[both_na & eq_or_na].reset_index(drop=True)

    #start a dataframe to build
    total = pd.DataFrame()

    #index of each dataframe to maintain
    idx = 1
    idy = 1

    #while there are items to fill into result DataFrame, loop
    while not cross.empty:
        
        #there are 3 possible next values
        #there is a matched word
        tmpn = cross[cross.word_x == cross.word_y]
        #x column doesn't have a match
        tmpx = cross[(cross.key_x == idx) & (cross.word_y.isna())].iloc[0:1]
        #y column doesn't have a match
        tmpy = cross[(cross.key_y == idy) & (cross.word_x.isna())].iloc[0:1]

        #special case in case a column has more data at the end
        if tmpn.empty:
            #tack on the next row from one or the other dataframe,
            #defer to the left side
            if tmpx.empty:
                tmp = tmpy[0:1]
                idy += 2
            else:
                tmp = tmpx[0:1]
                idx += 2
        #check if there is a word that comes before the next match in the left dataframe
        elif tmpn.iloc[0].key_x > tmpx.iloc[0].key_x + 1:
            idx += 2
            tmp = tmpx
        #check if there is a word that comes before the next match in the right dataframe
        elif tmpn.iloc[0].key_y > tmpy.iloc[0].key_y + 1:
            tmp = tmpy
            idy += 2
        #just use the next match
        else:
            tmp = tmpn.iloc[0:1]
            idx += 2
            idy += 2

        #build up the result table
        total = pd.concat([total, tmp])
        #remove the elements that have been used up
        cross = cross[(cross.key_x >= idx - 1) & (cross.key_y >= idy - 1)]

    #remove and rename intermediate columns
    total[['word_x', 'word_y']] = total[['word_x', 'word_y']].fillna(method='bfill', axis=1)
    total.rename({'word_x':'word'}, axis=1, inplace=True)
    return total.drop(['word_y', "key_x", "key_y"], axis = 1).reset_index(drop=True)

not_parathesis = re.compile(r"[^\(]+")
word_punctuation = re.compile(r"[\w']+|[.,!?;]")


def getEarlyStartTime(transcript_start):
    return max(transcript_start - PAD_TIME, 0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='Table Maker',
                    description='Makes a single table from all data sources')

    parser.add_argument('-t', '--transcript', help=f"Input single word CSV likely from TranscriptToWord, default={TRANSCRIPTS}", default=TRANSCRIPTS)
    parser.add_argument('-a', '--audio', help=f"Input single word CSV likely from TranscriptToWord, default={AUDIO_PATH}", default=AUDIO_PATH)
    parser.add_argument('-o', '--output', help=f"CSV to output combined table as, default={WORD_TRANSCRIPT}", default=WORD_TRANSCRIPT)

    args = parser.parse_args()

    with open(Path(args.transcript), 'r') as f:
        data = json.loads(f.read())

    table = buildTable(data, Path(args.audio))

    table.to_csv(args.output)


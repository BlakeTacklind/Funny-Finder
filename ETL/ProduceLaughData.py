#!/usr/bin/env python
# coding: utf-8

import torch, sys, os, numpy as np
from functools import partial
from tqdm import tqdm
import json
import argparse

from pathlib import Path

sys.path.append('laughter_detection/utils/')
sys.path.append('laughter_detection/')
import configs, data_loaders
import audio_utils

AUDIO_PATH = "audio/"
MODEL_PATH = 'laughter_detection/checkpoints/in_use/resnet_with_augmentation'

SAMPLE_RATE = 8000

#
def getPredictedValues(modelPath, audioPath):
    model, config = getModel(modelPath)

    audioFiles = [x for x in audioPath.iterdir() if not x.name.endswith('part')]

    values = [{
        'id': file.name.split('.')[0],
        'values':getLaughValues(model, file, config),
        "length":audio_utils.get_audio_length(file)
    } for file in audioFiles]

    return values

#from model, a single file, and configuration get the laugh data time slice
def getLaughValues(model, path, config):
    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=path, feature_fn=config['feature_fn'], sr=SAMPLE_RATE)

    collate_fn=partial(audio_utils.pad_sequences_with_labels,
                            expand_channel_dim=config['expand_channel_dim'])

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset, num_workers=4, batch_size=8, shuffle=False, collate_fn=collate_fn)

    probs = []
    for model_inputs, _ in tqdm(inference_generator):
        x = torch.from_numpy(model_inputs).float().to(getDevice())
        preds = model(x).cpu().detach().numpy().squeeze()
        if len(preds.shape)==0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds

    return probs

#returns a string for using the gpu or cpu
def getDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load the luaghter model
def getModel(model_path):

    config = configs.CONFIG_MAP['resnet_with_augmentation']

    device = getDevice()

    if not torch.cuda.is_available():
        print("using CPU, GPU highly recommneded")

    model = config['model'](dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])

    model.set_device(device)

    if os.path.exists(model_path):
        load_checkpoint(model_path+'/best.pth.tar', model, gpu = torch.cuda.is_available())
        model.eval()
    else:
        raise Exception(f"Model checkpoint not found at {model_path}")

    return model, config



#pulled from torch_utils but modified to take gpu as an argument
def load_checkpoint(checkpoint, model, optimizer=None, gpu=False):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint

    Modified from: https://github.com/cs230-stanford/cs230-code-examples/
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    else:
        print("Loading checkpoint at:", checkpoint)

    if gpu:
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if 'epoch' in checkpoint:
        model.epoch = checkpoint['epoch']

    if 'global_step' in checkpoint:
        model.global_step = checkpoint['global_step'] + 1
        print("Loading checkpoint at step: ", model.global_step)

    if 'best_val_loss' in checkpoint:
        model.best_val_loss = checkpoint['best_val_loss']

    return checkpoint



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='Produce Laughter Data',
                    description='Produce laughter probabilities from audio data')

    parser.add_argument('-i', '--input', help=f"Path to directory with Audio Files, deafult='{AUDIO_PATH}'", default=AUDIO_PATH)
    parser.add_argument('-o', '--output', help="File to output data to json, deafult='laughs.json'", default='laughs.json')

    args = parser.parse_args()

    jsonValues = getPredictedValues(MODEL_PATH, Path(args.input))

    with open(parser.output, 'w') as f:
        f.write(json.dumps(jsonValues))






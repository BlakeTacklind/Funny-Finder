#!/usr/bin/env python
# coding: utf-8

# # Using tf Dataset + Bucket Datasets + Hyperparam Tuning
# 
# Using tensorflow dataset from generator, significantly reduces memory overhead

import pandas as pd, numpy as np
from pathlib import Path
import tensorflow as tf
import keras_tuner as kt
from keras import backend as K
import os
import argparse
import sys

DATA_SET = "trainable.csv"
GRID_MODEL_DIR='grid_training'
SAVE_LOC = os.path.join('..','Webpage','predictor','models')

#training parameters
TRAINING_BUCKETS = 5
MAX_TRIALS = 50
MAX_EPOCHS = 100
CHECKPOINT_FILEPATH = '.mdl_wts.hdf5'

def Train(dataFile, target, modelName):

    train, test, val = buildDataSet(dataFile, target)

    tunned = runSearch(train, val, modelName)

    return buildBestModel(tunned, train, val, test)

#Build model search space
def runSearch(train, validation, modelName):

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max')
    lruPlateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        patience=5,
        mode='max',
    )

    tuner = kt.BayesianOptimization(model_builder,
                         objective=kt.Objective('val_auc', 'max'),
                         directory=GRID_MODEL_DIR,
                         max_trials=MAX_TRIALS,
                         project_name=modelName)


    tuner.search(train, epochs=MAX_EPOCHS, validation_data=validation, callbacks=[early_stopping, lruPlateau])

    return tuner

#Set up model with hyper parameters
def model_builder(hp):

    #list of hyper parameters to tune
    LSTMSize = hp.Int('lstm_units', min_value=32, max_value=512, step=32)
    denseSize = hp.Int('dense_size', min_value=32, max_value=1024, step=32)
    denseLayers = hp.Int('dense_layers', min_value=1, max_value=3)
    drop_rate = 0
    # drop_rate = hp.Float('drop_rate', min_value=0, max_value=0.25)
    bidirection = hp.Boolean('bidirection')
    
    model2 = makeModel(LSTMSize, denseSize, denseLayers, drop_rate, bidirection)

    model2.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.AUC(name='auc')],
    )
    
    return model2

#build model from parameters
def makeModel(LSTMSize, denseSize, denseLayers, drop_rate, bidirectional):
    model = tf.keras.models.Sequential()
    
    #mask to ignore zero data, allows for batching
    model.add(tf.keras.layers.Masking(mask_value=0.0))
    
    #LSTM layer
    if bidirectional:
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(LSTMSize, return_sequences=True)
        ))
    else:
        model.add(tf.keras.layers.LSTM(LSTMSize, return_sequences=True))
    
    #add a series of dense layers with dropouts
    for i in range(denseLayers):
        model.add(tf.keras.layers.Dense(denseSize, activation = 'relu'))
        # model.add(tf.keras.layers.Dropout(drop_rate))

    #output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

### Section on Loading Dataset

#load and split data
def buildDataSet(dataFile, target):

    data = loadData(dataFile, target)

    def getGen():
        return data

    ds = tf.data.Dataset.from_generator(getGen, output_types=(tf.float32, tf.int32), output_shapes=((None,300),(None, 1)))

    train, test, val = getDatasetPartitions(ds, len(data), shuffle=True)

    train_b = bucketTable(train)
    test_b = bucketTable(test)
    val_b = bucketTable(val)

    return train_b, test_b, val_b

#Load data from file into an array
def loadData(datafile, target):
    df = pd.read_csv(datafile, index_col=['id', 'number'])
    embed = [i for i in df.columns if i.startswith("embed")]
    
    #this conversion keeps tensor from complaining about the combining types
    #in a single tensor table
    df[target] = df[target].astype('float')
    all_tensor = df[embed+[target]].groupby('id').apply(tf.convert_to_tensor)
    
    #return a list of tuples for input and output values
    return [(tf.cast(val[:,:-1], tf.float32), tf.cast(val[:,-1:], tf.int32)) for _,val in all_tensor.items()]


#create buckets for use in improving training batches
def bucketTable(table, buckets = TRAINING_BUCKETS):
    #this is bad but unclear how to do it otherwise
    sortedList = sorted([len(x) for x,y in table])
    
    boundries = [sortedList[int(i * len(sortedList)/buckets)] for i in range(1, buckets)]
    boundrieSizes = [int((i + 1) * len(sortedList)/buckets) - int(i * len(sortedList)/buckets) for i in range(0, buckets)]
    
    def lengthList(elem, elem2):
        return tf.shape(elem)[0]

    return table.bucket_by_sequence_length(
            element_length_func=lengthList,
            bucket_boundaries=boundries,
            bucket_batch_sizes=boundrieSizes)

#Split by videos
def getDatasetPartitions(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12, reshuffle_each_iteration=False)
    
    #find a split so there is a relatively enven number of rows
    v = tf.math.cumsum(list(ds.map(lambda x,y: len(x))))
    high_value = tf.get_static_value(v[-1])
    low_count = high_value * train_split
    mid_count = high_value * (train_split + val_split)

    train_size = np.sum(v < low_count)
    val_size = np.sum(v < mid_count) - train_size

    if train_size == 0 or val_size == 0 or np.sum(v >= mid_count) == 0:
        raise Exception(f"Somehow got 0 sized section.\
Total Size {ds_size}, Train Size {train_size}, Validation Size {val_size}, Test Size {np.sum(v >= mid_count)}")
    
    #create generator subsets of dataset
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


#from a tunned set of hyper parameters build and test the model
def buildBestModel(tunner, train, validation, test, showResults=True):

    mcp_save = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_FILEPATH, save_best_only=True, monitor='val_auc', mode='min')

    lruPlateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        patience=5,
        mode='max',
    )

    #build a model
    best_hp = tunner.get_best_hyperparameters()[0]
    model = tunner.hypermodel.build(best_hp)

    history = model.fit(train, epochs=MAX_EPOCHS, validation_data=validation, callbacks=[mcp_save, lruPlateau])

    #load best found data
    model.load_weights(CHECKPOINT_FILEPATH)

    #print results of model
    if showResults:
        print(model.summary())
        print(model.evaluate(test))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Train',
                    description='Train on input data to produce 2 different models for usage in front end')

    parser.add_argument('-i', '--input', help=f"Input CSV likely from TableMaker, default={DATA_SET}", default=DATA_SET)
    parser.add_argument('-o', '--output', help=f"Diretory to save models in, default={SAVE_LOC}", default=SAVE_LOC)
    parser.add_argument('-g', '--gpu', help="Flag to require the GPU for training, default=False", default=False, action='store_true')

    args = parser.parse_args()


    if len(tf.config.list_physical_devices('GPU')) < 1:
        if args.gpu:
            print("Requires using a GPU", file=sys.stderr)
            exit(1)

        print("Using CPU, Super Slow")

    #create a model for each case
    for modelName, target in [("model1", "cata"), ("model0", "cata2")]:
        model = Train(Path(args.input), target, modelName)
        model.save(os.path.join(args.output, modelName))

        #GPU seems to need clean up afterwards
        K.clear_session()


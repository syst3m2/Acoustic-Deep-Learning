# %%
# import argparse
import calendar
import datetime
import glob
import json
import math
import os
import pprint
import random
import signal
import time

import numpy as np
#import sqlalchemy as db
import tensorflow as tf
#import tensorflow_datasets as tfds
from sklearn.preprocessing import MultiLabelBinarizer
#from vs_data_query import WavCrawler

#from resources.acoustic_pipeline import *
#from resources.ais_pipeline import *

def parse_tfr_element(args, element, output_data):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'seq_num' : tf.io.FixedLenFeature([], tf.int64),
        'samples': tf.io.FixedLenFeature([], tf.int64),
        'channels':tf.io.FixedLenFeature([], tf.int64),
        'sample_rate':tf.io.FixedLenFeature([], tf.int64),
        'audio' : tf.io.FixedLenFeature([], tf.string),
        'label':tf.io.FixedLenFeature([], tf.string),
        'label_len':tf.io.FixedLenFeature([], tf.int64),
        'start_time':tf.io.FixedLenFeature([], tf.int64),
        'end_time':tf.io.FixedLenFeature([], tf.int64),
        'mmsi/lat/long/brg/rng' : tf.io.FixedLenFeature([], tf.string),
        'positions_len' : tf.io.FixedLenFeature([], tf.int64),
        'positions_width' : tf.io.FixedLenFeature([], tf.int64)
        }
        
    content = tf.io.parse_single_example(element, data)
    
    seq_num = content['seq_num']
    samples = content['samples']
    channels = content['channels']
    sample_rate = content['sample_rate']
    audio = content['audio']
    label = content['label']
    label_len = content['label_len']
    start_time = content['start_time']
    end_time = content['end_time']
    positions = content['mmsi/lat/long/brg/rng']
    positions_len = content['positions_len']
    positions_width = content['positions_width']
    
    #get our feature and reshape it appropriately
    feature = tf.io.parse_tensor(audio, out_type=tf.float64)
    feature = tf.reshape(feature, shape=[samples,channels])

    label = tf.io.parse_tensor(label, out_type=tf.int64)
    label = tf.reshape(label, shape=[label_len])

    positions = tf.io.parse_tensor(positions, out_type=tf.string)
    positions = tf.reshape(positions, shape=[positions_len,positions_width])

    if output_data=='features':
        return (feature, label)
    elif output_data=='metadata':
        return (feature, label, seq_num, samples, channels, sample_rate, start_time, end_time, positions, label_len, positions_len, positions_width)

def parse_mel_tfr_element(args, element, output_data):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'seq_num' : tf.io.FixedLenFeature([], tf.int64),
        'length': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channels':tf.io.FixedLenFeature([], tf.int64),
        'sample_rate':tf.io.FixedLenFeature([], tf.int64),
        'audio' : tf.io.FixedLenFeature([], tf.string),
        'label':tf.io.FixedLenFeature([], tf.string),
        'label_len':tf.io.FixedLenFeature([], tf.int64),
        'start_time':tf.io.FixedLenFeature([], tf.int64),
        'end_time':tf.io.FixedLenFeature([], tf.int64),
        'mmsi/lat/long/brg/rng' : tf.io.FixedLenFeature([], tf.string),
        'positions_len' : tf.io.FixedLenFeature([], tf.int64),
        'positions_width' : tf.io.FixedLenFeature([], tf.int64)
        }
        
    content = tf.io.parse_single_example(element, data)
    
    seq_num = content['seq_num']
    length = content['length']
    width = content['width']
    channels = content['channels']
    sample_rate = content['sample_rate']
    audio = content['audio']
    label = content['label']
    start_time = content['start_time']
    end_time = content['end_time']
    positions = content['mmsi/lat/long/brg/rng']
    label_len = content['label_len']
    positions_len = content['positions_len']
    positions_width = content['positions_width']
    
    #get our feature and reshape it appropriately

    feature = tf.io.parse_tensor(audio, out_type=tf.float64)
    feature = tf.reshape(feature, shape=[length,width,channels])

    # Add option if number of channels is 1
    output_channels = args.channels
    if output_channels==1:
        feature = feature[:,:,0]
        feature = tf.expand_dims(feature, axis=2)

    label = tf.io.parse_tensor(label, out_type=tf.int64)
    label = tf.reshape(label, shape=[label_len])

    positions = tf.io.parse_tensor(positions, out_type=tf.string)
    positions = tf.reshape(positions, shape=[positions_len,positions_width])

    if output_data=='features':
        return (feature, label)
    elif output_data=='metadata':
        return (feature, label, seq_num, length, width, channels, sample_rate, start_time, end_time, positions, label_len, positions_len, positions_width)
    elif output_data=='features-times':
        return (feature, label, start_time, end_time)

# This function reads tfrecords in from a directory into a TF Dataset
# file_directory: Firectory holding tf records to read into dataset
# output_type: raw, calibrated, mel, calibrated-mel
# output_data: features or metadata. Features for feature and label, metadata for all other data
def get_audio_dataset(args, files, batch_size, output_type, output_data):

    dataset = tf.data.TFRecordDataset(files,num_parallel_reads=16)

    #pass every single feature through our mapping function
    if output_type == 'mel' or output_type == 'calibrated-mel':
        dataset = dataset.map(
            lambda x: parse_mel_tfr_element(args, x, output_data)
        )    
    else:
        dataset = dataset.map(
            lambda x: parse_tfr_element(args, x, output_data)
        )
        
    #dataset = dataset.batch(batch_size,drop_remainder=True)

    meta_files = []
    for record in files:
        json_file = record.replace(".tfrecords", ".json")
        meta_files.append(json_file)


    multi_class_count = {}
    multi_label_count = {}
    example_count = 0

    for fname in meta_files:

        f = open(fname)
        
        metadata = json.load(f)
        ml_count = metadata['MultiLabel Class Count']
        mc_count = metadata['Multiclass Class Count']
        example_count += metadata['Examples Count']
        for key in ml_count:
            if key in multi_label_count:
                multi_label_count[key] += ml_count[key]
            else:
                multi_label_count[key] = ml_count[key]

        for key in mc_count:
            if key in multi_class_count:
                multi_class_count[key] += mc_count[key]
            else:
                multi_class_count[key] = mc_count[key]

        f.close()

    dataset_metadata = {"Examples Count":example_count, "MultiLabel Class Count": multi_label_count, "Multiclass Class Count":multi_class_count}
    
    return dataset, dataset_metadata

def split_dataset(json_dir,seed,test_pct=10,val_pct=20,train_pct=70):
    random.seed(seed)
    """
    :param json_dir: the path to the folder containing the json files with tfrecord dataset metadata
    :param test_pct: the percentage of records to place in the test set
    :param val_pct: the percentage of records to place in the validation set
    :param train_pct: the percentage of records to place in the training set

    reads the json files created with the tfrecords in order to create splits
    
    the input json file structure is as follows:
    {
        "Examples Count": 64,
        "MultiLabel Class Count": {
            "Class A,Class D": 64
        },
        "Multiclass Class Count": {
            "Class A": 64,
            "Class D ": 64
        },
        "Channels ": 4,
        "Sample Rate ": 4000,
        "Label Range ": 20,
        "Start Time ": "20210806 - 232935",
        "End Time ": "20210806 - 235342"
    }
    the output json file ("data_splits.json") has keys for the multi-label classes 
    and for the test, validation, and training data splits. the multi-label class 
    keys identify how many tfrecords are included for given class and the tfrecord 
    files which represent them
    {
        "a_c" : {
            "count" : 2389,
            "files" : [
                "file1.tfrecords",
                "file2.tfrecords"
            ]
        },
        "b" : {
            "count" : 123,
            "files" : [
                "file3.tfrecords",
                "file1.tfrecords"
            ]
        }
    }
    """
    tot_json = 0
    ml_class_files = {}
    for _file in glob.glob("{}/*.json".format(json_dir)):
        if "data_splits.json" in _file:
            continue
        with open(os.path.join(json_dir,_file)) as tf_json:
            metadata = json.load(tf_json)
            for key in metadata["MultiLabel Class Count"].keys():
                # format the key from Class A,Class C => a_c, where Class C,Class A is also a_c
                ml_labels = key.lower().replace("class","").replace(" ","").split(",")
                sorted(ml_labels)
                _k = "_".join(ml_labels)
                if _k not in ml_class_files:
                    ml_class_files[_k] = { "count" : 0, "files" : [] }
                ml_class_files[_k]["count"] = ml_class_files[_k]["count"] + 1
                ml_class_files[_k]["files"].append(_file.replace(".json",".tfrecords"))
                # this creates entries in the json file representing the multi-label classes 
                # and the files which include examples containing them

    # shuffle the file lists for each of the ml classes
    for k in ml_class_files.keys():
        ml_class_files[k]["files"] = \
            random.sample(ml_class_files[k]["files"], len(ml_class_files[k]["files"]))

    # with the file lists shuffled, generate the test/validation/training datasets
    ml_test_vald_train = {"test_set":[],"validation_set":[],"train_set":[]}
    for k in ml_class_files.keys():
        ml_ds = ml_class_files[k]

        test_count = (test_pct/100) * ml_ds["count"]
        test_count = math.floor(test_count)
        _test_list = ml_ds["files"][0:test_count-1]
        ml_test_vald_train["test_set"].extend(_test_list)
        
        vald_count = (val_pct/100) * ml_ds["count"]
        vald_count = math.floor(vald_count)
        _vald_list = ml_ds["files"][test_count:test_count+vald_count-1]
        ml_test_vald_train["validation_set"].extend(_vald_list)

        train_count = (train_pct/100) * ml_ds["count"]
        train_count = math.floor(train_count)
        _train_list = ml_ds["files"][vald_count:vald_count+train_count-1]
        ml_test_vald_train["train_set"].extend(_train_list)

    ml_class_files = {**ml_class_files, **ml_test_vald_train}

    print("\n###################\ntest_set size: {}\nvalidation_set size: {}\ntrain_set size: {}\n###################\n"
        .format(len(ml_class_files["test_set"]),len(ml_class_files["validation_set"]),len(ml_class_files["train_set"])))

    data_splits_json = json.dumps(ml_class_files)

    cwd = os.getcwd()
    #file_directory = os.path.join(os.path.relpath(cwd,json_dir),'data_splits.json')
    file_directory = os.path.join(json_dir,'data_splits_sabrina.json')
    '''
    with open(file_directory, "w") as outfile:
        json.dump(data_splits_json, outfile)  
    '''
    return ml_class_files

if __name__ == '__main__':

    # If you need to create a split dataset, use this code
    file_directory_absolute = '/smallwork/beards/CS4321/acoustic_datasets/mel_calibrated_052018_082021_v2'
    data_splits = split_dataset(json_dir=file_directory_absolute,seed=4321)


    # If you just want to read in the dataset from a json file, then use this
    '''
    json_file = os.path.join(file_directory_absolute, "data_splits.json")
    f = open(json_file)
    data_splits = json.load(f)
    '''

    test_files = data_splits['test_set']
    validation_files = data_splits['validation_set']
    train_files = data_splits['train_set']

    print("Creating Dataset")
    train_dataset, train_metadata = get_audio_dataset(train_files, 256, 'calibrated-mel', 'features')
    test_dataset, test_metadata = get_audio_dataset(test_files, 256, 'calibrated-mel', 'features')
    val_dataset, val_metadata = get_audio_dataset(validation_files, 256, 'calibrated-mel', 'features')

    print("\n###################\nTrain Dataset Metadata\n###################\n")
    print(train_metadata)
    print("\n###################\nTest Dataset Metadata\n###################\n")
    print(test_metadata)
    print("\n###################\nVal Dataset Metadata\n###################\n")
    print(val_metadata)
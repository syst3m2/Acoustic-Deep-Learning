# This script contains a pipeline to generate a dataset with raw physics audio data and AIS .mat files
# Environment on cluster: audio_pipeline
from data_pipeline.resources.ais_pipeline import *
from data_pipeline.resources.acoustic_pipeline import *
from vs_data_query import WavCrawler
import argparse
import os
import sqlalchemy as db
#import tensorflow_datasets as tfds
import tensorflow as tf
import datetime
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import glob
import time 
import calendar
import signal
import json
import sys

# Sample rate of the raw audio data is 8000
# this is to define the segment length, but will be downsampled to 4000 in the data processing pipeline
WC_SAMPLE_RATE = 8000

ship_class_dict ={'Landings Craft':'Class A', 'Military ops':'Class A','Fishing vessel':'Class A','Fishing Vessel':'Class A' ,'Fishing Support Vessel':'Class A', 'Tug':'Class A', 'Pusher Tug':'Class A', 'Dredging or UW ops':'Class A', 'Towing vessel':'Class A', 'Crew Boat':'Class A', 'Buoy/Lighthouse Vessel':'Class A', 'Salvage Ship':'Class A', 'Research Vessel':'Class A', 'Anti-polution':'Class A', 'Offshore Tug/Supply Ship':'Class A', 'Law enforcment':'Class A', 'Landing Craft':'Class A', 'SAR':'Class A', 'Patrol Vessel':'Class A', 'Pollution Control Vessel': 'Class A', 'Offshore Support Vessel':'Class A',
                        'Pleasure craft':'Class B', 'Yacht':'Class B', 'Sailing vessel':'Class B', 'Pilot':'Class B', 'Diving ops':'Class B', 
                        'Passenger (Cruise) Ship':'Class C', 'Passenger Ship':'Class C', 'Passenger ship':'Class C', 'Training Ship': 'Class C',
                        'Naval/Naval Auxiliary':'Class D','DDG':'Class D','LCS':'Class D','Hospital Vessel':'Class D' ,'Self Discharging Bulk Carrier':'Class D' ,'Cutter':'Class D', 'Passenger/Ro-Ro Cargo Ship':'Class D', 'Heavy Load Carrier':'Class D', 'Vessel (function unknown)':'Class D',
                        'General Cargo Ship':'Class D','Wood Chips Carrier':'Class D', 'Bulk Carrier':'Class D' ,'Cement Carrier':'Class D','Vehicles Carrier':'Class D','Cargo ship':'Class D', 'Oil Products Tanker':'Class D', 'Ro-Ro Cargo Ship':'Class D', 'USNS RAINIER':'Class D', 'Supply Tender':'Class D', 'Cargo ship':'Class D', 'LPG Tanker':'Class D', 'Crude Oil Tanker':'Class D', 'Container Ship':'Class D', 'Container ship':'Class D','Bulk Carrier':'Class D', 'Chemical/Oil Products Tanker':'Class D', 'Refrigerated Cargo Ship':'Class D', 'Tanker':'Class D', 'Car Carrier':'Class D', 'Deck Cargo Ship' :'Class D', 'Livestock Carrier': 'Class D',
                        'Bunkering Tanker':'Class D', 'Water Tanker': 'Class D', 'FSO': 'Class D', 'Towing vessel (tow>200)':'Class A', 'Hopper Dredger':'Class C', 'Cable Layer':'Class D', 'Fish Carrier':'Class A', 'LNG Tanker':'Class D',
                        'not ship':'Class E' }

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value is tensor
        #print(value)
        #input("A tensor made its way into the bytes feature, press ENTER to continue")
        value = value.numpy() # get value of tensor

    #if isinstance(value,list) or isinstance(value,np.ndarray):
    #    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(str(value), 'utf-8')]))
    #else:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    #try:
    if isinstance(array,list):
        array = np.array(array)

    array = tf.io.serialize_tensor(array)
    #except ValueError:
    #    print("There was a value error serializing the tensor due to mixed types")
    #    print(array)
    #    print(type(array))
    #    input("Observe array, press ENTER to continue")
        
    #print(array)
    return array

def date_to_utc(date_str, precision='day'):
    """Converts a date string in the format MM/DD/YYYY to a UTC timestamp

    Parameters
    ----------
    date_str : str
        the date string in the appropriate formate MM/DD/YYYY

    Returns
    ----------
    number 
        the UTC timestamp as a number
    """
    #utc_time = time.mktime(datetime.datetime.strptime(date_str, "%m/%d/%Y").timetuple())
    if precision=='day':
        utc_time = calendar.timegm(datetime.datetime.strptime(date_str, '%Y%m%d').timetuple())
    elif precision=='second':
        utc_time = calendar.timegm(datetime.datetime.strptime(date_str, "%Y%m%d %H%M%S").timetuple())
    else:
        print("Please specify a precision for start time and end time, day or second", flush=True)
    return utc_time


# Use for parsing raw or calibrated data
def parse_audio(audio, label, start_time, end_time, sample_rate, seq_number, positions):

    # Save multi instance and lat/long/brg/range as additional features
    #define the dictionary -- the structure -- of our single example
    #positions = bytes(str(positions), 'utf-8')

    if isinstance(positions,list):
        positions = np.array(positions)

    

    data = {
            'seq_num' : _int64_feature(seq_number),
            'samples' : _int64_feature(audio.shape[0]),
            'channels' : _int64_feature(audio.shape[1]),
            'sample_rate' : _int64_feature(sample_rate),
            'audio' : _bytes_feature(serialize_array(audio)),
            'label' : _bytes_feature(serialize_array(label)),
            'label_len' : _int64_feature(len(label)),
            'start_time' : _int64_feature(start_time),
            'end_time' : _int64_feature(end_time),
            'mmsi/lat/long/brg/rng' : _bytes_feature(serialize_array(positions)),
            'positions_len' : _int64_feature(positions.shape[0]),
            'positions_width' : _int64_feature(positions.shape[1])
        }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

# Use this for parsing mel spectrogram version because it is a different shape
def parse_mel_audio(audio, label, start_time, end_time, sample_rate, seq_number, positions):

    if isinstance(positions,list):
        print("Converting list to np array", flush=True)
        positions = np.array(positions)
    
    data = {
            'seq_num' : _int64_feature(seq_number),
            'length' : _int64_feature(audio.shape[0]),
            'width' : _int64_feature(audio.shape[1]),
            'channels' : _int64_feature(audio.shape[2]),
            'sample_rate' : _int64_feature(sample_rate),
            'audio' : _bytes_feature(serialize_array(audio)),
            'label' : _bytes_feature(serialize_array(label)),
            'label_len' : _int64_feature(len(label)),
            'start_time' : _int64_feature(start_time),
            'end_time' : _int64_feature(end_time),
            'mmsi/lat/long/brg/rng' : _bytes_feature(serialize_array(positions)),
            'positions_len' : _int64_feature(positions.shape[0]),
            'positions_width' : _int64_feature(positions.shape[1])
        }

    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def tfr_audio_writer(audio, labels, timestamps, sample_rate, filename, output_data, id_counter, positions):
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    counter = 0

    for index in range(len(audio)):

        #get the data we want to write
        current_audio = audio[index] 
        current_label = labels[index]
        current_start_time = timestamps[index][0]
        current_end_time = timestamps[index][1]
        current_id = id_counter[index]
        current_positions = np.array(positions[index])

        
        #print(current_positions)
        #print(type(current_positions))
        if output_data == 'mel' or output_data == 'calibrated-mel':
            out = parse_mel_audio(audio=current_audio, label=current_label, sample_rate=sample_rate, start_time=current_start_time, end_time=current_end_time, seq_number=current_id, positions=current_positions)
        else:
            out = parse_audio(audio=current_audio, label=current_label, sample_rate=sample_rate, start_time=current_start_time, end_time=current_end_time, seq_number=current_id, positions=current_positions)

        writer.write(out.SerializeToString())
        counter += 1

    writer.close()
    print(f"Wrote {counter} elements to TFRecord", flush=True)
    
    return counter

def parse_tfr_element(element, output_data):
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

def parse_mel_tfr_element(element, output_data):
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

    label = tf.io.parse_tensor(label, out_type=tf.int64)
    label = tf.reshape(label, shape=[label_len])

    positions = tf.io.parse_tensor(positions, out_type=tf.string)
    positions = tf.reshape(positions, shape=[positions_len,positions_width])

    if output_data=='features':
        return (feature, label)
    elif output_data=='metadata':
        return (feature, label, seq_num, length, width, channels, sample_rate, start_time, end_time, positions, label_len, positions_len, positions_width)

# This function reads tfrecords in from a directory into a TF Dataset
# file_directory: Firectory holding tf records to read into dataset
# output_type: raw, calibrated, mel, calibrated-mel
# output_data: features or metadata. Features for feature and label, metadata for all other data
def get_audio_dataset(file_directory, batch_size, output_type, output_data):
    #create the dataset
    # 

    cwd = os.getcwd()
    file_directory = os.path.join(os.path.relpath(cwd,file_directory),file_directory)

    # Create the dataset
    files = glob.glob(os.path.join(file_directory,"*.tfrecords"))
    #files = os.listdir(file_directory)

    #dataset_size = len(files) * 64

    dataset = tf.data.TFRecordDataset(files)

    #pass every single feature through our mapping function
    if output_type == 'mel' or output_type == 'calibrated-mel':
        dataset = dataset.map(
            lambda x: parse_mel_tfr_element(x, output_data)
        )    
    else:
        dataset = dataset.map(
            lambda x: parse_tfr_element(x, output_data)
        )
        
    dataset = dataset.batch(batch_size)

    # Get the metadata for the dataset
    meta_files = glob.glob(os.path.join(file_directory,"*.json"))
    
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
'''
def get_audio_dataset(file_directory, batch_size, output_type, output_data):
    #create the dataset
    # 

    cwd = os.getcwd()
    file_directory = os.path.join(os.path.relpath(cwd,file_directory),file_directory)
    files = glob.glob(os.path.join(file_directory,"*.tfrecords"))
    #files = os.listdir(file_directory)

    dataset = tf.data.TFRecordDataset(files)

    #pass every single feature through our mapping function
    if output_type == 'mel' or output_type == 'calibrated-mel':
        dataset = dataset.map(
            lambda x: parse_mel_tfr_element(x, output_data)
        )    
    else:
        dataset = dataset.map(
            lambda x: parse_tfr_element(x, output_data)
        )
        
    dataset = dataset.batch(batch_size)
    
    return dataset
'''

def multi_label_one_hot_encoder(labels):
    """ 
        This function transforms the list of labels into a one hot encoded vector
        The list is the list of all labels in the dataset
        Uses sklearn encoders. For the multilabel case

        Returns: 2D list in one hot form, or for multi-label case, multi-hot 
    """
    onehot_encoder = MultiLabelBinarizer(classes=['Class A', 'Class B', 'Class C', 'Class D', 'Class E'])
    onehot_vector = onehot_encoder.fit_transform(labels)

    return onehot_vector

def sig_handler(signum, frame):
    print("Segfault occurred", flush=True)
    print("Trying again", flush=True)


def main(length, output_data, ais_folder, audio_database, output_folder, channels, label_range, sample_rate, cwd, start_date, end_date, label_type):
    
    start = time.perf_counter()
    #if not os.path.isabs(ais_folder):
    #    ais_folder = os.path.join(cwd, ais_folder)
    #if not os.path.isabs(audio_database):
    #    audio_database = os.path.join(cwd, audio_database)
    #if not os.path.isabs(output_folder):
    #    output_folder = os.path.join(cwd, output_folder)

    audio_database = os.path.join(audio_database,'master_index.db')
    # Get time range of entire dataset from audio database
    #if not audio_folder.endswith('/'):
    #    audio_folder = audio_folder + '/'

    ais_folder = os.path.join(os.path.relpath(cwd,ais_folder),ais_folder)
    output_folder = os.path.join(os.path.relpath(cwd,output_folder),output_folder)

    audio_database = os.path.join(os.path.relpath(cwd,audio_database),audio_database)
    #print(audio_database)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    #audio_database = 'sqlite:////' + audio_database

    # Process for getting .mat ais data
    # Get ais data
    print("Loading AIS data", flush=True)
    print("Loading MMSIs", flush=True)
    #print(ais_folder)

    #mat_start_file = datetime.datetime.strptime(start_date, "%Y%m%d")
    mat_start_file = datetime.datetime.strptime(start_date, "%Y%m%d %H%M%S")
    mat_start_file = mat_start_file.date()
    mat_start_file = mat_start_file - datetime.timedelta(days=1)
    #mat_start_file = mat_start_file.strftime("%y%m%d")
    #mat_start_file = mat_start_file + ".mat"

    #mat_end_file = datetime.datetime.strptime(end_date, '%Y%m%d')
    mat_end_file = datetime.datetime.strptime(end_date, '%Y%m%d %H%M%S')
    mat_end_file = mat_end_file.date()
    #mat_end_file = mat_end_file - datetime.timedelta(days=1)
    #mat_start_file = mat_start_file.strftime("%y%m%d")
    #mat_start_file = mat_start_file + ".mat"

    mat_files = glob.glob(os.path.join(ais_folder,'*.mat'))
    #mat_files = os.listdir(ais_folder)
    
    ais_file_list = []
    for filename in mat_files:
        file_date = filename.split("/")[-1][:-4]
        #file_date = filename[:-4]
        file_date = datetime.datetime.strptime(file_date, "%y%m%d")
        file_date = file_date.date()
        if file_date >= mat_start_file and file_date <= mat_end_file:
            ais_file_list.append(os.path.join(ais_folder,filename))


    position_range = label_range + 100

    #all_mmsis = gen_data_per_file(dir=ais_file_list, cpa_rng_max=label_range+20, cpa_rng_min=0)
    all_mmsis = gen_data_per_file(dir=ais_file_list, cpa_rng_max=position_range, cpa_rng_min=0)
    
    print("Scraping web for additional data", flush=True)
    mmsi_db = mmsi_scraper(mmsis=all_mmsis)

    # Creates dataframe of all the ships in the dataset, and their start and end times for transiting within range of the sensor
    #all_ships = generate_multi_label_ships(data_dir=ais_folder, mmsiDB_file=mmsi_db, far_rng=label_range+20, max_rng=label_range, min_rng=0.0)
    #all_ships['SHIP_CLASS'] = all_ships['DESIG'].apply(lambda x: [ship_class_dict[desig] if desig in ship_class_dict else 'NaN' for desig in x])

    print("Formatting AIS data", flush=True)
    # Returns a dataframe of ships within desired range, and ships dataframe for the further range to determine times to count as Class E
    # Returns a dataframe to use ship positions
    all_ships, all_ships_beyond, positions_df = generate_single_output(data_dir=ais_file_list, mmsiDB=mmsi_db, cpa_rng_beyond=label_range+10, cpa_rng_max=label_range,position_range=position_range, cpa_rng_min=0.0)

    # Make the same dataframe but to get ship positions at a greater range
    #position_df, position_beyond_df = generate_single_output(data_dir=ais_file_list, mmsiDB=mmsi_db, cpa_rng_beyond=position_range+20, cpa_rng_max=position_range, cpa_rng_min=0.0)


    #all_mmsis = gen_data_per_file(dir=ais_file_list, cpa_rng_max=label_range+20, cpa_rng_min=0)
    #mmsi_db = mmsi_scraper(mmsis=all_mmsis)
    #all_ships, all_ships_beyond = generate_single_output(data_dir=ais_file_list, mmsiDB=mmsi_db, cpa_rng_beyond=label_range+20, cpa_rng_max=label_range, cpa_rng_min=0.0)

    all_ships['SHIP_CLASS'] = all_ships['DESIG'].map(ship_class_dict)
    all_ships_beyond['SHIP_CLASS'] = all_ships_beyond['DESIG'].map(ship_class_dict)
    positions_df['SHIP_CLASS'] = positions_df['DESIG'].map(ship_class_dict)

    print("AIS data loaded!", flush=True)

    csv_name = start_date.replace("/","")

    all_mmsis.to_csv(os.path.join(output_folder,csv_name+'_all_mmsis.csv'),index=False)
    mmsi_db.to_csv(os.path.join(output_folder,csv_name+'_mmsi_db.csv'),index=False)
    all_ships.to_csv(os.path.join(output_folder,csv_name+'_mmsi_times.csv'),index=False)


    # Need a process for getting ais data from database

    ###########################

    # Get the time range of the dataset t1-t2
    '''
    acoustic_engine = db.create_engine(audio_database)
    acoustic_connection = acoustic_engine.connect()
    latest_time_query = "SELECT MAX(end_time_sec) FROM AUDIO"
    time2 = acoustic_connection.execute(latest_time_query)
    t2 = time2.all()[0][0]

    earliest_time_query = "SELECT MIN(start_time_sec) FROM AUDIO"
    time1 = acoustic_connection.execute(earliest_time_query)
    t1 = time1.all()[0][0]

    #t2 = t1+3600

    # Close database connections
    acoustic_connection.close()
    acoustic_engine.dispose()
    '''

    t1 = date_to_utc(start_date, precision='second')
    t2 = date_to_utc(end_date, precision='second')

    # Based on the sample rate of 8000 and desired audio clip length, determine segment length
    segment_length = length * WC_SAMPLE_RATE

    audio_start_time = datetime.datetime.utcfromtimestamp(t1).strftime("%Y%m%d-%H%M%S")
    audio_end_time = datetime.datetime.utcfromtimestamp(t2).strftime("%Y%m%d-%H%M%S")

    #audio_start_time = datetime.datetime.utcfromtimestamp(utc_start_time).strftime("%Y%m%d-%H%M%S")
    #audio_end_time = datetime.datetime.utcfromtimestamp(utc_end_time).strftime("%Y%m%d-%H%M%S")

    # If there is no corresponding value in the dictionary, save those desigs to a file. 
    # Can use to edit the ship class dictionary later
    # Drop ships not in dictionary
    no_desig = all_ships[all_ships['SHIP_CLASS'].isna()]
    #This is from pandas, returns all ships with NA ship class column
    if not no_desig.empty:
        print("There are ships without a mapped designation, check na_desigs.csv to add these to the dictionary. Check avg_dwt_per_class.csv to determine appropriate class labels", flush=True)
        no_desig = mmsi_db[mmsi_db['MMSI'].isin(no_desig['MMSI'].tolist())]
        no_desig.to_csv(os.path.join(output_folder, 'na_desigs_'+audio_start_time+'_'+audio_end_time+'.csv'), index=False)

        # Save down average dead weight per class to appropriately categorize no designation data
        mmsi_db['SHIP_CLASS'] = mmsi_db['DESIG'].map(ship_class_dict)
        mmsi_db['DWT'] = mmsi_db['DWT'].replace('-',-1)
        mmsi_db['DWT'] = mmsi_db['DWT'].astype('int')

        # Get rid of the -1 values, because it messes with the average
        avg_mmsi_db = mmsi_db[mmsi_db['DWT']!=-1]

        grouped_db = avg_mmsi_db.groupby('SHIP_CLASS').mean().reset_index()
        grouped_db.to_csv(os.path.join(output_folder, 'avg_dwt_per_class_'+audio_start_time+'_'+audio_end_time+'.csv'), index=False)

    all_ships = all_ships[all_ships['SHIP_CLASS'].notna()]

    # Process the acoustic data

    print("Total audio dataset from " + audio_start_time + " to " + audio_end_time, flush=True)

    # Using time range, create wavcrawler object
    wc = WavCrawler(audio_database, t1, t2, segment_length=segment_length, overlap=0.25)

    # Iterate through wavcrawler, save metadata and audio data to list
    # Save lists as tfrecords once there are 100 data points
    timestamp_list = []
    audio_list = []
    true_labels_list = []
    count=0
    id_counter = 0
    id_counter_list = []
    positions_list = []
    no_ais_list = []

    true_label_dict = {}
    multi_class_true_label_dict = {}

    DISCARD_FLAG = False
    for segment in wc:
        #seg=True

        # Grab timestamp from segment, grab date, look for ais file with same date. If does not exist, then skip
        times = [x for x in segment.time_stamp]
        segment_time = datetime.datetime.utcfromtimestamp(times[0]).date()
        segment_time = segment_time.strftime("%y%m%d")
        ais_file_lookup = segment_time + ".mat"
        
        file_path_lookup = os.path.join(ais_folder,ais_file_lookup)
        
        # If file exists, then do nothing, if not, then continue
        if not os.path.exists(file_path_lookup):
            
            if segment_time not in no_ais_list:
                print("AIS file for " + segment_time + " does not exist, skipping. Check no_data.txt for a list of dates that are not in the dataset", flush=True)
                no_ais_list.append(segment_time)

            continue

        # If a segfault occurs, then try again until success
        #signal.signal(signal.SIGSEGV, sig_handler)
        #while seg:
        if output_data=='raw':
            dataset, timestamps = wavcrawler_data_process(segment=segment, mode='single', channels=channels, segment_dur=length, calibrate=False, sample_rate=4000)

        elif output_data=='calibrated':
            dataset, timestamps = wavcrawler_data_process(segment=segment, mode='single', channels=channels, segment_dur=length, calibrate=True, sample_rate=4000)
        
        elif output_data=='mel':
            dataset, timestamps = full_mel_mfcc_pipeline(segment, channels=channels, mode='single', source='wc', segment_dur=length, calibrate=False, sample_rate=4000)

        elif output_data=='calibrated-mel':
            dataset, timestamps = full_mel_mfcc_pipeline(segment, channels=channels, mode='single', source='wc', segment_dur=length, calibrate=True, sample_rate=4000)
        
        #seg=False

        # Functions return tensors, so convert to list
        dataset = dataset.numpy().tolist()

        audio_start_time = datetime.datetime.utcfromtimestamp(timestamps[0])
        audio_end_time = datetime.datetime.utcfromtimestamp(timestamps[1])

        print("Processing data from " + audio_start_time.strftime("%Y%m%d-%H%M%S") + " to " + audio_end_time.strftime("%Y%m%d-%H%M%S"), flush=True)

        # Grab the start time of the file for the filename
        if count==0:
            file_start_time = audio_start_time.strftime("%Y%m%d-%H%M%S")

        # Get the true labels for that time segment

        # Use this if generate_single_output function used
        #print(all_ships.dtypes)
        #print(all_ships_beyond.dtypes)
        true_labels = all_ships[ \
                ((all_ships['END_TIME']>=audio_start_time) & (all_ships['END_TIME']<=audio_end_time)) | \
                ((all_ships['START_TIME']>=audio_start_time) & (all_ships['START_TIME']<=audio_end_time)) | \
                ((all_ships['START_TIME']<=audio_start_time) & (all_ships['END_TIME']>=audio_end_time)) | \
                ((all_ships['START_TIME']>=audio_start_time) & (all_ships['END_TIME']<=audio_end_time))]['SHIP_CLASS'].tolist()

        true_labels_classe = all_ships_beyond[((all_ships_beyond['END_TIME']>=audio_start_time) & (all_ships_beyond['END_TIME']<=audio_end_time)) | \
                  ((all_ships_beyond['START_TIME']>=audio_start_time) & (all_ships_beyond['START_TIME']<=audio_end_time)) | \
                  ((all_ships_beyond['START_TIME']<=audio_start_time) & (all_ships_beyond['END_TIME']>=audio_end_time)) | \
                    ((all_ships_beyond['START_TIME']>=audio_start_time) & (all_ships_beyond['END_TIME']<=audio_end_time))]['SHIP_CLASS'].tolist()

        '''
        positions = all_ships[((all_ships['END_TIME']>=audio_start_time) & (all_ships['END_TIME']<=audio_end_time)) | \
                  ((all_ships['START_TIME']>=audio_start_time) & (all_ships['START_TIME']<=audio_end_time)) | \
                  ((all_ships['START_TIME']<=audio_start_time) & (all_ships['END_TIME']>=audio_end_time))]['MMSI,TIME,LAT,LON,BRG,RNG'].apply(pd.Series).stack().reset_index(drop=True).tolist()
        
        positions = all_ships[((all_ships['END_TIME']>=audio_start_time) & (all_ships['END_TIME']<=audio_end_time)) | \
                  ((all_ships['START_TIME']>=audio_start_time) & (all_ships['START_TIME']<=audio_end_time)) | \
                  ((all_ships['START_TIME']<=audio_start_time) & (all_ships['END_TIME']>=audio_end_time)) | \
                  ((all_ships['START_TIME']>=audio_start_time) & (all_ships['END_TIME']<=audio_end_time))]['MMSI,TIME,LAT,LON,BRG,RNG']
        '''

        positions = positions_df[((positions_df['END_TIME']>=audio_start_time) & (positions_df['END_TIME']<=audio_end_time)) | \
                  ((positions_df['START_TIME']>=audio_start_time) & (positions_df['START_TIME']<=audio_end_time)) | \
                  ((positions_df['START_TIME']<=audio_start_time) & (positions_df['END_TIME']>=audio_end_time)) | \
                  ((positions_df['START_TIME']>=audio_start_time) & (positions_df['END_TIME']<=audio_end_time))]['MMSI,TIME,LAT,LON,BRG,RNG,DESIG']

        
        # Save lat/long positions for that time period
        #audio_start_time_hour = audio_start_time - datetime.timedelta(hours=1)
        #audio_end_time_hour = audio_end_time + datetime.timedelta(hours=1)
        if len(positions) > 0:
            positions = positions.apply(pd.Series).stack().reset_index(drop=True).tolist()
            positions_tmp = pd.DataFrame(positions)

            # Find the ship class for the positions
            positions_tmp[7] = positions_tmp[6].map(ship_class_dict)

            #positions_df[1] = pd.to_datetime(positions_df[1], format='%m/%d/%Y-%H:%M:%S')
            #print(positions)
            #print(positions_df)
            #print(positions_df.columns)
            #print(positions_df.dtypes)
            #input("Positions dataframe")
            #positions_df = positions_df[(positions_df[1]>=audio_start_time) & (positions_df[1]<=audio_end_time)]
            #print(positions_df[1].max())
            #print(positions_df[1].min())
            positions = positions_tmp.to_numpy().tolist()

        elif len(positions)==0:
            positions = [['0','0','0','0','0','0','0','0']]
        
        new_positions = []
        for pos in positions:
            if None in pos:
                continue
            else:
                new_positions.append(pos)
                
        positions = new_positions

        # Use this if generate_multilabel_ship function used 
        # Creates list of lists of labels
        '''
        true_labels = all_ships[((all_ships['END_TIME']>=audio_start_time) & (all_ships['END_TIME']<=audio_end_time)) | \
                  ((all_ships['START_TIME']>=audio_start_time) & (all_ships['START_TIME']<=audio_end_time)) | \
                  ((all_ships['START_TIME']<=audio_start_time) & (all_ships['START_TIME']>=audio_end_time))]['SHIP_CLASS'].tolist()

        true_labels = [item for sublist in true_labels for item in sublist]
        '''

        '''
        for idx in range(len(positions)):
            if None in positions[idx]:
                ['0' None None None None None]
        '''

        # When there are ships between the ranges, don't discard the data. We will still want it for plotting and
        # prediction experiments
        # Instead, flag the tfrecord file to be discarded in training
        if len(true_labels_classe)==0:
            true_labels=['Class E']
            #print("There are no true labels for this time period (ships only present between no ship range and label range), discarding data")
            #continue
        elif len(true_labels)==0 and len(true_labels_classe)>0:
            #print("There are no true labels for this time period (ships only present between no ship range and label range), discarding data")
            print("There are ships only present between no ship range and label range, labeling file with discard flag", flush=True)
            print("These files will be marked as Class E, but should be discarded during training", flush=True)
            true_labels=['Class E']
            DISCARD_FLAG=True
            #continue

        
        

        # If desired dataset is multi class, then discard instances with multiple ships present
        # Else if multi label, then keep unique instances of ships

        if label_type == 'multiclass':
            if len(true_labels)>1:
                print("There are multiple ships present, labeling file with discard flag, but will keep labels for data, should not be used for training multilabel models", flush=True)
                DISCARD_FLAG=True
                #continue

        #class_dict = {"Class A":0, "Class B":1, "Class C":2, "Class D":3, "Class E":4}
        # Since it is multi-label and not multi-instance, we will need to only keep unique instances of each class
        # If a multi-instance dataset is needed, get rid of this line for true labels
        true_labels = list(set(true_labels))

        # Sort the list to ensure uniformity
        priority = {"Class A":0, "Class B":1, "Class C":2, "Class D":3, "Class E":4}

        true_labels = sorted(true_labels, key=priority.get)

        true_label_string = ','.join(true_labels)
        if true_label_string in true_label_dict:
            true_label_dict[true_label_string] += 1
        else:
            true_label_dict[true_label_string] = 1

        for item in true_labels:
            if item in multi_class_true_label_dict:
                multi_class_true_label_dict[item] += 1
            else:
                multi_class_true_label_dict[item] = 1
            

        '''
        label_filename = ""
        for lab in true_labels:
            # remove Class
            # [A, B]
            # A_B
            label_filename = label_filename + "_" + str(lab)
        '''

        # There is no good way to do multi instance with this code
        #true_labels_multi_instance = np.array(list(true_labels))

        # Convert to numbers for each class
        #true_labels = [class_dict[x] for x in true_labels]

        # Perform multi-label binarization for labels

        #print(positions)
        # Append all data to list
        timestamp_list.append(timestamps)
        audio_list.append(dataset)
        true_labels_list.append(true_labels)
        positions_list.append(positions)
        #positions_list = np.append(positions_list,positions)
        id_counter_list.append([id_counter])

        count += 1
        id_counter += 1

        print("Processing " + str(count) + " of 64", flush=True)
        print("Processing id number " + str(id_counter), flush=True)

        sys.stdout.flush()
        
        # Save batches of 64 to each TF record
        if count==64: #64:
            timestamps_np = np.array(timestamp_list)

            true_labels_np = multi_label_one_hot_encoder(true_labels_list)
            audio_np = np.array(audio_list)
            positions_np = positions_list
 
            
            id_counter_np = np.array(id_counter_list)
            #print(positions_np)
            
            file_end_time = audio_end_time.strftime("%Y%m%d-%H%M%S")

            if DISCARD_FLAG:
                filename = output_data + "_" + str(channels) + "channel_" + str(sample_rate) + "sr_" + label_type + '_' + str(label_range) + "km_" + str(count) + "count_" + file_start_time + "_" + file_end_time + "_" + "discard"
                DISCARD_FLAG=False
            else:
                filename = output_data + "_" + str(channels) + "channel_" + str(sample_rate) + "sr_" + label_type + '_' + str(label_range) + "km_" + str(count) + "count_" + file_start_time + "_" + file_end_time + "_" + "include"

            print("Saving tfrecord " + filename, flush=True)

            filepath = os.path.join(output_folder,filename+".tfrecords")

            # Write audio data to tfrecord
            num_file_records = tfr_audio_writer(audio_np, true_labels_np, timestamps_np, sample_rate, filepath, output_data, id_counter_np, positions_np)

            # Write class count metadata to json

            metadata_dict = {"Examples Count":count, "MultiLabel Class Count": true_label_dict, "Multiclass Class Count":multi_class_true_label_dict,"Channels":channels,"Sample Rate": sample_rate, "Label Range": label_range, "Start Time": file_start_time, "End Time": file_end_time}
            class_count_filename = os.path.join(output_folder,filename+".json")

            with open(class_count_filename, "w") as outfile:
                json.dump(metadata_dict, outfile)

            true_label_dict = {}
            multi_class_true_label_dict = {}


            # reset lists
            count=0
            timestamp_list = []
            audio_list = []
            true_labels_list = []
            positions_list = []
            #positions_list = np.array([])
            id_counter_list = []



    # If loop exited and count is greater than 0, then there is more data to write out
    
    if count > 0:
        timestamps_np = np.array(timestamp_list)
        true_labels_np = multi_label_one_hot_encoder(true_labels_list)
        #true_labels_np = np.array(true_labels_list)
        audio_np = np.array(audio_list)
        positions_np = positions_list
        #positions_np = np.array(positions_list)
        #positions_np = positions_np.astype('object')
        #print(positions_np)
        #print(type(positions_np))
        id_counter_np = np.array(id_counter_list)

        file_end_time = audio_end_time.strftime("%Y%m%d-%H%M%S")

        if DISCARD_FLAG:
            filename = output_data + "_" + str(channels) + "channel_" + str(sample_rate) + "sr_" + label_type + '_' + str(label_range) + "km_" + str(count) + "count_" + file_start_time + "_" + file_end_time + "_" + "discard"
            DISCARD_FLAG=False        
        else:
            filename = output_data + "_" + str(channels) + "channel_" + str(sample_rate) + "sr_" + label_type + '_' + str(label_range) + "km_" + str(count) + "count_" + file_start_time + "_" + file_end_time + "_" + "include"


        #filename = output_data + "_" + str(channels) + "channel_" + str(sample_rate) + "sr_" + label_type + '_' + str(label_range) + "km_" + str(count) + "count_" + file_start_time + "_" + file_end_time

        print("Saving final tfrecord " + filename, flush=True)

        filepath = os.path.join(output_folder,filename+".tfrecords")

        # Write audio data to tfrecord
        num_file_records = tfr_audio_writer(audio_np, true_labels_np, timestamps_np, sample_rate, filepath, output_data, id_counter_np, positions_np)

        # Write class count metadata to json

        metadata_dict = {"Examples Count":count, "MultiLabel Class Count": true_label_dict, "Multiclass Class Count":multi_class_true_label_dict,"Channels":channels,"Sample Rate": sample_rate, "Label Range": label_range, "Start Time": file_start_time, "End Time": file_end_time}
        class_count_filename = os.path.join(output_folder,filename+".json")

        with open(class_count_filename, "w") as outfile:
            json.dump(metadata_dict, outfile)

        true_label_dict = {}
        multi_class_true_label_dict = {}


    # Save .txt file with all dates that were not created
    message = "Data was not available for the following dates due to a lack of AIS data\n"

    no_datafile = open(os.path.join(output_folder, 'no_data.txt'), 'w+')

    no_datafile.write(message)
    for date in no_ais_list:
        no_datafile.write(date + "\n")

    no_datafile.close()

    
    # At this point, all data in the dataset should be written out as tfrecords with the appropriate labels
    # In the output folder path
    stop = time.perf_counter()
    print("Dataset created!", flush=True)
    total_time = stop - start

    print("Took " + str(total_time//60) + " minutes to complete dataprocessing", flush=True)



if __name__ == "__main__":
    
    cwd = os.getcwd()

    parser = argparse.ArgumentParser()

    parser.add_argument('--length',       type=int,   default=30,                 help="number of seconds to generate the audio for, each labeled audio will correspond to this number")
    parser.add_argument('--output_data',  type=str,   default='mel',              help="'raw','calibrated', 'mel', or 'calibrated-mel'. Output either raw audio data, calibrated audio data, or the mel spectrogram of the audio data")
    parser.add_argument('--ais_folder',   type=str,                               help="absolute file path containing ais .mat files")
    parser.add_argument('--audio_database', type=str,                             help="file path containing audio database (Absolute filepath is required, do not include filename or leading '/')")
    parser.add_argument('--output_folder',type=str,   default='dataset_output',   help="absolute file path for desired data output to save to")
    parser.add_argument('--channels',     type=int,   default=4,                  help="1 or 4. Number of channels to have in the resulting dataset")
    parser.add_argument('--label_range',  type=int,   default=20,                 help="Range in kilometers to count as true labels for ships")
    parser.add_argument('--sample_rate',  type=int,   default=4000,               help="Desired sample rate of resulting dataset to downsample from original (less than  or equal to 8000)")
    parser.add_argument('--label_type',  type=str,   default='multilabel',        help="Indicate whether you would like a multi class or a multi-label dataset")
    parser.add_argument('--start_date',   type=str,   default='01/01/2000',       help="The date for which the first audio file should be gathered")
    parser.add_argument('--end_date',     type=str,   default='12/31/2030',       help="The date for which the last audio file should be gathered (not inclusive; e.g. 10/01/2021 will collect up until but not including that date)")

    args = parser.parse_args()

    main(args.length, args.output_data, args.ais_folder, args.audio_database, args.output_folder, args.channels, args.label_range, args.sample_rate, cwd, args.start_date, args.end_date, args.label_type)

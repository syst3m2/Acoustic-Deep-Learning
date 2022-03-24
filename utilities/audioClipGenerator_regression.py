''' 
    Audio Clip Generator
    Andrew Pfau
    Sonar Classifier work

    This program converts data stored in excel or csv files into lists of samples that can be used by the data/dataset_processor.py to form the 
    data input pipeline. There are several stages to this process including data extraction from other files, applying labels to samples, and converting
    hour long audio files into 30 second sample files. Functions usually save output in a csv file in the same folder for use by the next function, this 
    aids in troubleshooting issues with file naming and time generation. 
    
    UPDATES: Updated to include bearing and range to target ship from sensor. Bearing and range can now be added as labels for regression inference

    Functions contained within this file are:

    from_excel_file: This function reads in data stored in an excel file to produce a csv file for use in later functions. Can process data stored in
    multiple sheets of the same workbook file

    audioClipper: Cuts hour long audio files into shorter files of 30 seconds. Uses the audioClipper_multi_proc function to take advantage of multi-processing
    to speed up this process

    make_samples: Function performs the logical conversion of hour long files into many 30 second sample files

    make_time_segments_harp: Function takes in listings of AIS contacts and generates the file names where the audio comes from and applies labels to the files
    based on ship properties retrieved from MMSI information.

    make_time_segments_phys: Performs the same functions as make_time_segments_harp but for physics data. These functions are sperate due to file name convention and
    alternate methods of generating labels for the two datasets

    create_hour_files: Concatonates audio files into hour long files. Assumes that 3 different files are required to generate 1 hour long file. Uses
    concat_files_multi_proc to do this with multiple processes to be more efficient

    main: starting point for program, contains commadline arguments and data paths

    Data Flow Paths:
    Functions are used in the following order to create a folder containing samples
    AIS Contact CSV file -> make_time_segments_x -> make_samples -> audioClipper

    This program is based on the audioClipGenerator.py file but has been modified to operate on the physics data and file formats

'''
# library imports
import pandas as pd
import os
import datetime
import argparse
import numpy as np
import glob
import concurrent.futures
import sys
import pickle

#for audio functions
#import librosa
import scipy.signal as scipy_signal
import copy
import soundfile as sf


def interpolate(x_1, x_2, y_1, y_2, interval):
    """
    Interpolate between 2 points for both x and y values, y_1 must be first time value and match x_1,
    Args:
        x_1, x_2 - ranges values to interpolate between
        y_1, y_2 - time values to interpolate between
        interval - the time steps desired to interpolate the x value for in seconds

    Returns:
        interpolated_data - list of interpolated x values
    """ 
    interpolated_data = []
    
    # determine if subtracting or adding to interpolate
    count_down = False
    if x_1 > x_2:
        count_down = True
    
    num_intervals = (y_2-y_1).seconds 
    # km or bearing per second, need to use abs
    x_interval = abs(x_1 - x_2) / num_intervals
    interpolated_data.append(x_1)
    # start plus x_interval * num seconds from x_1 to target
    for step in range(0, num_intervals): 
        if count_down:
            interpolated_data.append(x_1 - (x_interval * step))
        else:
            interpolated_data.append(x_1 + (x_interval * step))

    interpolated_data.append(x_2)
    return interpolated_data

class CPA_OBJECT(object):
    """
    Object to handle repeated operations for cpa multi labeling
    """
    def __init__(self, cpa_labels, hr=None, min=None):
        self.cpa_labeling = cpa_labels
        self.cpa_hr = hr
        self.cpa_min = min
        self.end_of_hr = datetime.time(0, 59, 59)
        self.start_of_hr = datetime.time(0,0,0)

    def set_time(self, hr, min):
        self.cpa_hr = hr
        self.cpa_min = min

    def get_cpa_min(self, current_hr):
        cpa_time = 0
        if self.cpa_labeling:
            if current_hr == self.cpa_hr:
                cpa_time = self.cpa_min
            elif current_hr < self.cpa_hr:
                cpa_time = self.end_of_hr
            elif current_hr > self.cpa_hr:
                cpa_time = self.start_of_hr

        return cpa_time


def get_class(label):
    """
    Function to serve as a central location for class labeling info
    label -> one ship desig_listnator
    return respective class
    """
    ship_class_dict ={'Landings Craft':'classA', 'Military ops':'classA','Fishing vessel':'classA','Fishing Vessel':'classA' ,'Fishing Support Vessel':'classA', 'Tug':'classA', 'Pusher Tug':'classA', 'Dredging or UW ops':'classA', 'Towing vessel':'classA', 'Crew Boat':'classA', 'Buoy/Lighthouse Vessel':'classA', 'Salvage Ship':'classA', 'Research Vessel':'classA', 'Anti-polution':'classA', 'Offshore Tug/Supply Ship':'classA', 'Law enforcment':'classA', 'Landing Craft':'classA', 'SAR':'classA', 'Patrol Vessel':'classA', 'Pollution Control Vessel': 'classA', 'Offshore Support Vessel':'classA', 'Hopper Dredger':'classA',
                        'Pleasure craft':'classB', 'Yacht':'classB', 'Sailing vessel':'classB', 'Pilot':'classB', 'Diving ops':'classB', 
                        'Passenger (Cruise) Ship':'classC', 'Passenger Ship':'classC', 'Passenger ship':'classC', 'Training Ship': 'classC',
                        'Naval/Naval Auxiliary':'classD','DDG':'classD','LCS':'classD','Hospital Vessel':'classD' ,'Self Discharging Bulk Carrier':'classD' ,'Cutter':'classD', 'Passenger/Ro-Ro Cargo Ship':'classD', 'Heavy Load Carrier':'classD', 'Vessel (function unknown)':'classD',
                        'General Cargo Ship':'classD','Wood Chips Carrier':'classD', 'Bulk Carrier':'classD' ,'Cement Carrier':'classD','Vehicles Carrier':'classD','Cargo ship (HAZ-A)':'classD', 'Oil Products Tanker':'classD', 'Ro-Ro Cargo Ship':'classD', 'USNS RAINIER':'classD', 'LPG Tanker':'classD', 'Crude Oil Tanker':'classD', 'Container Ship':'classD', 'Bulk Carrier':'classD', 'Chemical/Oil Products Tanker':'classD', 'Refrigerated Cargo Ship':'classD', 'Tanker':'classD', 'Car Carrier':'classD', 'Deck Cargo Ship' :'classD', 'Livestock Carrier': 'classD',
                        'Bunkering Tanker':'classD', 'Water Tanker': 'classD', 'FSO': 'classD', 'Supply Tender' : 'classD', 
                        'not ship':'classE' }
    if label in ship_class_dict:
        return ship_class_dict[label]
    else:
        print("Label: " + str(label) + " not assigned to a class")
        return ""

def from_excel_file(excel_filename):
    """
     This method takes the master excel file and transforms it into a csv file of filename, start time, end time, label
     label come from the sheet in the excel file that lists the audio clip
     Arguments: excel_filename: name of .xls file located in the same directory that contains the data
    """

    cwd = os.getcwd()
    # list of all sheets to be read in
    listOfSheets = ['SHIP', 'GRAY', 'Fin', 'Delphinid sp', 'Humpbacks']
    LABEL_DICT = {'SHIP': "ship", 'GRAY':"whale", 'Fin':"whale",'Delphinid sp':"dolphin", 'Humpbacks':"whale"}
    allData = pd.read_excel(os.path.join(cwd, excel_filename), sheet_name = listOfSheets, header = 0)
    sliceData = pd.DataFrame(columns=["FILE_NAME", "START_TIME", "END_TIME", "LABEL", "MMSI", "DESIG"])
    for sheet in allData:
        for row in allData[sheet].iterrows():
            # only need last 2 digits
            #41334.13556134259 breaks on this in dolphin sheet
            #print(row[1]['DTG_START'])
            yr = str(row[1]['DTG_START'].year)[2:]
            day = row[1]['DTG_START'].day
            label = LABEL_DICT[sheet]
            if day < 10:
                day = '0' + str(day)
            else:
                day = str(day)
            month = row[1]['DTG_START'].month
            if month < 10:
                month = '0'+str(month)
            else: 
                month = str(month)
            hr = row[1]['DTG_START'].hour
            if hr < 10:
                hr = '0' + str(hr) # add 0 to front of hour
            else:
                hr = str(hr)

            startMin = datetime.time(0, row[1]['DTG_START'].minute, row[1]['DTG_START'].second)
            # for segments that cross hours
            if hr != row[1]['DTG_END'].hour:
                endMin = datetime.time(0, 59, 59)
                # file for first hr
                extractFile = 'PacFLT_TM01_'+ yr + month+ day+ "_" + hr +"0000.d10.x.wav"
                sliceData = sliceData.append({'FILE_NAME':extractFile, 'START_TIME':startMin, 'END_TIME':endMin, 'LABEL':label}, ignore_index = True)
                # file for second hr
                startMin = datetime.time(0,0,0)
                endMin = datetime.time(0, row[1]['DTG_END'].minute, row[1]['DTG_END'].second)
                hr2 = row[1]['DTG_END'].hour
                if hr2 < 10:
                    hr2 = '0' + str(hr2)
                else:
                    hr2= str(hr2)
                extractFile = 'PacFLT_TM01_'+ yr + month+ day+ "_" + hr2 +"0000.d10.x.wav"
                sliceData = sliceData.append({'FILE_NAME':extractFile, 'START_TIME':startMin, 'END_TIME':endMin, 'LABEL':label}, ignore_index = True)
            else:
                endMin = datetime.time(0, row[1]['DTG_END'].minute, row[1]['DTG_END'].second)
                extractFile = 'PacFLT_TM01_'+ yr + month + day+ "_" + hr +"0000.d10.x.wav"
                sliceData = sliceData.append({'FILE_NAME':extractFile, 'START_TIME':startMin, 'END_TIME':endMin, 'LABEL':label}, ignore_index = True)

    # write out to csv
    sliceData.to_csv(os.path.join(cwd,'tmp/excel_labels.csv'), index=False)


def audioClipper_multi_proc(file_entry, data_folder, dest_folder, s_rate, dur):
    """
    This function is called by audioClipper in mapping process so that it is multi processed
    """
    # columns in csv file of the following info
    FILE_NAME = 0
    START_TIME = 1
    LABELS = 3
    
    # this is the unique part of the filename
    filename = file_entry[1][FILE_NAME].split('.')[0]
    # split on underscore
    filehead = filename.split('_')

    # convert filename format to operate on HARP data filenames
    # there is a slight difference in the files saved as 4k downsampled and all other files for this dataset
    if filehead[0] == 'PacFLT':
        filename = file_entry[1][FILE_NAME].replace('_', '-')
        filename = filename.replace('TM01', 'DL12')
    else:
        filename = file_entry[1][FILE_NAME]

    #labelName = file_entry[1][LABELS]
    labelName = ""
    for l in file_entry[1][LABELS].split(','):
        if l != '':
            labelName += l + "-"
    # time in seconds of where to start clippling
    offset = (file_entry[1][START_TIME].minute*60) + (file_entry[1][START_TIME].second)
    # open audio file and clip
    if os.path.exists(os.path.join(data_folder, filename)):
        # read in 30 seconds of audio
        audioData = sf.read(os.path.join(data_folder, filename), start=(offset * s_rate) , stop =((offset+dur) * s_rate))
        # save new audio clip in uniquely labeled file format:
        # LABEL_DATE_HOUR_INDEX.wav
        # EX: ship_121201_01_123.wav 
        newAudioFile = labelName + "_" + filehead[2] + "_" + filehead[3][:2] + "_" + str(offset) + ".wav"
        sf.write(os.path.join(dest_folder, newAudioFile), audioData[0], samplerate = s_rate)
        
    else:
        print('File: ' +str(filename)+ ' not found')

def audioClipper(csvFile, data_folder, dest_folder, s_rate, duration):
    """
        This function reads in a csv file containing files to be divided up, the start and end time of the clips, and the label for the file
        The assumed format for the file names is : mars_data_YYYYMMDD_HR.wav
        Params:
        csvFile: file containing data described above, assumed to be located in the current directory
        data_folder: folder containing full length audio files
        dest_folder: folder to put new audio clips in, must exist already
        s_rate: sample rate of full length auido, will be the same rate as the output audio, no down or up sampling is performed here
    
        Output: csv of newfilepaths and associated labels
    """
    max_cpu = os.cpu_count()-1 
    labelFile = pd.read_csv(os.path.join(os.getcwd(), csvFile), parse_dates = [1,2], header = 0)    
    # make lists for passing args to map
    data_folder_list = [data_folder] * len(labelFile)
    dest_folder_list = [dest_folder] * len(labelFile)
    dur_list = [duration] * len(labelFile)
    sr_list = [s_rate] * len(labelFile)
    fnames = []
    labels = []
    # extract file name and timestamp
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cpu) as executor:
        for fname in executor.map(audioClipper_multi_proc, labelFile.iterrows(), data_folder_list, dest_folder_list, sr_list, dur_list):
            pass

def make_samples(csv_file, dur):
    """ 
     This function reads in the full list of filenames and labels  divide each clip into it's 30 sec blocks
     perform the matching search above to match clips of the same times with their tags
     then arrange the list by order of file name for audio clip generation efficiency
     save one csv of just ships, save one of everything else
    """
    cwd = os.getcwd()
    full_list = pd.read_csv(os.path.join(cwd, csv_file), parse_dates = [1,2], header = 0)
    new_df = pd.DataFrame(columns=["FILE_NAME", "START_TIME", "END_TIME", "LABEL"])
    START = 1
    END = 2
    LABEL_LIST = 3
    NAME = 0
    deltaT = datetime.timedelta(seconds=dur)
    
    for clip in full_list.iterrows():
        # cut into segments
        clip_time = clip[1][END] - clip[1][START] 
        start = clip[1][START]
        for t in range((int)(clip_time.seconds / dur)):
            if (start + deltaT) <= clip[1][END]:
                new_df = new_df.append({'FILE_NAME':clip[1][NAME], 'START_TIME':start.time(), 'END_TIME':(start+deltaT).time(), 'LABEL':clip[1][LABEL_LIST]}, ignore_index=True)
            start += deltaT

    new_df.sort_values(by=['FILE_NAME'], inplace=True, ascending=False)
    new_df.to_csv('tmp/full_list_labels.csv', index=False)

def make_samples_cpa(csv_file, dur):
    """ 
     This function reads in the full list of filenames and labels  divide each clip into it's 30 sec blocks
     perform the matching search above to match clips of the same times with their tags
     then arrange the list by order of file name for audio clip generation efficiency
     save one csv of just ships, save one of everything else.
     Same as make_samples except it expects CPA time information in the csv_file and adds aspect to label list
    """
    cwd = os.getcwd()
    full_list = pd.read_csv(os.path.join(cwd, csv_file), parse_dates = [1,2,5], header = 0)
    new_df = pd.DataFrame(columns=["FILE_NAME", "START_TIME", "END_TIME", "LABEL"])
    START = 1
    END = 2
    LABEL_LIST = 3
    CPA_TIME = 5
    NAME = 0
    deltaT = datetime.timedelta(seconds=dur)
    
    for clip in full_list.iterrows():
        # cut into segments
        clip_time = clip[1][END] - clip[1][START] 
        start = clip[1][START]
        for t in range((int)(clip_time.seconds / dur)):
            if (start + deltaT) <= clip[1][END]:
                if (start + deltaT) <= clip[1][CPA_TIME]:
                    aspect = "closing"
                else:
                    aspect = "opening"    
                new_df = new_df.append({'FILE_NAME':clip[1][NAME], 'START_TIME':start.time(), 'END_TIME':(start+deltaT).time(), 'LABEL':clip[1][LABEL_LIST]+","+aspect}, ignore_index=True)

            start += deltaT

    new_df.sort_values(by=['FILE_NAME'], inplace=True, ascending=False)
    new_df.to_csv('full_list_labels.csv', index=False)

def make_samples_brg_rng(csv_file, dur, brg_flag, rng_flag):
    """ 
     This function reads in the full list of filenames and labels  divide each clip into it's 30 sec blocks
     perform the matching search above to match clips of the same times with their tags
     then arrange the list by order of file name for audio clip generation efficiency
     save one csv of just ships, save one of everything else
     Same as make_sample but add bearing and range calculations.
     Args:
        brg_flag: boolean flag from command line input, if true, bearing is added to label list
        rng_flag: boolean flag from command line input, if true, range is added to label list
    """
    cwd = os.getcwd()
    full_list = pd.read_csv(os.path.join(cwd, csv_file), parse_dates = [2,3], header = 0)
    # load data from pickle file
    # dict of all bearing and ranges in tuple (bearing, range)
    bearing_and_range_data = {}
    with open('tmp/bearing_range_obj.pkl', 'rb') as data_file: 
        bearing_and_range_data = pickle.load(data_file)
        data_file.close()
    
    new_df = pd.DataFrame(columns=["FILE_NAME", "START_TIME", "END_TIME", "LABEL"])
    MMSI  = 0
    NAME  = 1
    START = 2
    END   = 3
    LABEL_LIST = 4
    deltaT = datetime.timedelta(seconds=dur)

    # Option 1: calculate BRG/RNG for every second and then pick by sample time
    # option 2: try to calculate for sample times only
    for clip in full_list.iterrows():
        # cut into segments
        clip_time = clip[1][END] - clip[1][START] 
        start = clip[1][START]
        # get the sample date and hour time to find  
        clip_date = clip[1][NAME].split('_')
        hr = (int)(clip_date[3][:2])
        yr = (int)('20' + clip_date[2][:2])
        day = (int)(clip_date[2][4:])
        mon = (int)(clip_date[2][2:4])
   
        for t in range((int)(clip_time.seconds / dur)):
            label_list = clip[1][LABEL_LIST].split(',')
            brg = 0
            rng = 0
            mmsi = clip[1][MMSI]
            if mmsi != 0:
                # create a full date time object 
                sample_time = datetime.datetime(yr,mon,day,hr, start.minute, start.second)
                # get bearing and range for this sample time
                brg = bearing_and_range_data[mmsi][sample_time][0]
                rng = bearing_and_range_data[mmsi][sample_time][1]
            if (start + deltaT) <= clip[1][END]:
                # append bearing and range to label list if flags set
                if brg_flag and clip[1][MMSI] != 0:
                    label_list.append(round(brg,2))
                if rng_flag and clip[1][MMSI] != 0:
                    label_list.append(round(rng,2))
                new_df = new_df.append({'FILE_NAME':clip[1][NAME], 'START_TIME':start.time(), 'END_TIME':(start+deltaT).time(), 'LABEL':label_list}, ignore_index=True)
            start += deltaT

    new_df.sort_values(by=['FILE_NAME'], inplace=True, ascending=False)
    new_df.to_csv('tmp/full_list_labels.csv', index=False)

def overlapping_ships(csv_file):
    """
    This function takes the labels generated from AIS data and creates an output csv that is passed to make_time_segments
    Takes in a ship_list
    This function looks for overlapping ships and creates a new csv including overlapping time frames
    multi label 
    """
    cwd = os.getcwd()
    overlapping_data = pd.DataFrame(columns=["START_TIME", "END_TIME", "LABEL", "DESIG"]) # LABEL and DESIG can be lists
    allData = pd.read_csv(os.path.join(cwd, csv_file), parse_dates=[1,2], header = 0)
    start_time = 0
    end_time = 0
    for ship1 in allData.iterrows():
        for ship2 in allData.iterrows():
            if ship1[1]["START_TIME"] < ship2[1]["START_TIME"] < ship1[1]["END_TIME"]:
                start_time = ship2[1]["START_TIME"]
                if ship1[1]["END_TIME"] < ship2[1]["END_TIME"]:
                    end_time = ship1[1]["END_TIME"]
                else:
                    end_time = ship2[1]["END_TIME"]
                label = ship1[1]["LABEL"] + ";" + ship2[1]["LABEL"]
                desig_list = ship1[1]["DESIG"] + ";" + ship2[1]["DESIG"]
                overlapping_data = overlapping_data.append({"START_TIME": start_time, "END_TIME": end_time, "LABEL": label, "DESIG": desig_list}, ignore_index=True)
                # single label entries
                overlapping_data = overlapping_data.append({"START_TIME": ship1[1]["START_TIME"], "END_TIME": start_time, "LABEL": ship1[1]["LABEL"], "DESIG": ship1[1]["DESIG"]}, ignore_index=True)
                overlapping_data = overlapping_data.append({"START_TIME": end_time, "END_TIME": ship2[1]["END_TIME"], "LABEL": ship2[1]["LABEL"], "DESIG": ship2[1]["DESIG"]}, ignore_index=True)

    overlapping_data.to_csv(os.path.join(cwd,'ship_multi_label.csv'), index=False)


def make_time_segments(label_file, multi_label_flag, label_type, data_source, cpa_labels, pickled_file):
    """
    label_file: csv file containing data
    label_type: either class or mmsi for what type of labeling to generate
    data_source: either harp or mars for correct source file naming
    """
    cwd = os.getcwd()

    # classes based on ship's ear paper
    ctr =0
    file_start = ""
    file_ender = ""
    label = ""
    dates = [1,2]

    if cpa_labels:
        dates= [1,2,6]
    
    if pickled_file:
        #when reading in data with bearing and range it's easier to read in a pickled file than a csv
        if os.path.splitext(label_file)[1] != '.pkl':
            sys.exit("WARNING: Pickle file not given as input when attempting to load from pickle file, change csv_file to pickle file and try again")
        allData = pd.read_pickle(os.path.join(cwd, label_file))
    else:
        allData = pd.read_csv(os.path.join(cwd, label_file), parse_dates=dates, header = 0)
    
    sliceData = pd.DataFrame(columns=["MMSI", "FILE_NAME", "START_TIME", "END_TIME", "LABEL", "DESIG", "CPA_TIME"])
    brg_rng_dict = {}
    if data_source == 'harp':
        file_start = 'PacFLT_TM01_'
        file_ender = '0000.d10.x.wav'
    elif data_source == 'phys':
        file_start = 'mars_data_'
        file_ender = '.wav'
    
    for row in allData.iterrows():
        ctr += 1
        # only need last 2 digits
        label = ""
        mmsi = row[1]['MMSI']
        desig_list = row[1]['DESIG']
        if multi_label_flag:
            desig_list = desig_list.replace("'", '')
            desig_list = desig_list.replace('[', '')
            desig_list = desig_list.replace(']', '').strip()
            desig_list = desig_list.split(',')
        
        yr = str(row[1]['START_TIME'].year)[2:]
        day = row[1]['START_TIME'].day
        if day < 10:
            day = '0' + str(day)
        else:
            day = str(day)
        month = row[1]['START_TIME'].month
        if month < 10:
            month = '0'+str(month)
        else: 
            month = str(month)
        hr = row[1]['START_TIME'].hour
        if hr < 10:
            hr = '0' + str(hr) # add 0 to front of hour
        else:
            hr = str(hr)

        startMin = datetime.time(0, row[1]['START_TIME'].minute, row[1]['START_TIME'].second)

        cpa_obj = CPA_OBJECT(cpa_labels)
        if cpa_labels:
            cpa_hr = row[1]['CPA_TIME'].hour
            cpa_min = datetime.time(0, row[1]['CPA_TIME'].minute, row[1]['CPA_TIME'].second)
            cpa_obj.set_time(cpa_hr, cpa_min)

        # change output label as either directly from the AIS csv or based on the Designator
        # modify to account for multilabel case where desig is a list
        if label_type == 'class':
        # the label is based on the designator and ships are divided into classes
            if multi_label_flag:
                for desig in desig_list:
                    desig = desig.strip()
                    label += get_class(desig) + ','
            else:
                label = get_class(desig_list) 
        
        elif label_type == 'mmsi':
            label = row[1]['LABEL']

        # for segments that cross hours
        cpa_min = cpa_obj.get_cpa_min(row[1]['START_TIME'].hour )
        if row[1]['START_TIME'].hour != row[1]['END_TIME'].hour:
            endMin = datetime.time(0, 59, 59)
            # file for first hr start at 1st hour start min and end at 59:59
            extractFile = file_start+ yr + month+ day+ "_" + hr +file_ender
            
            cpa_min = cpa_obj.get_cpa_min(row[1]['START_TIME'].hour )
            
            sliceData = sliceData.append({'MMSI':mmsi,'FILE_NAME':extractFile, 'START_TIME':startMin, 'END_TIME':endMin, 'LABEL':label, 'DESIG':desig_list, 'CPA_TIME':cpa_min}, ignore_index = True)
            # file for second hr, start at 00:00 and end at 2nd hour min
            # the target recording only overlaps 1 hour
            if (row[1]['END_TIME'].hour == row[1]['START_TIME'].hour + 1):
                startMin = datetime.time(0,0,0)
                endMin = datetime.time(0, row[1]['END_TIME'].minute, row[1]['END_TIME'].second)
                hr2 = row[1]['END_TIME'].hour
                cpa_min = cpa_obj.get_cpa_min(hr2)
            
                if hr2 < 10:
                    hr2 = '0' + str(hr2)
                else:
                    hr2= str(hr2)
                extractFile = file_start+ yr + month+ day+ "_" + hr2 +file_ender
                sliceData = sliceData.append({'MMSI':mmsi, 'FILE_NAME':extractFile, 'START_TIME':startMin, 'END_TIME':endMin, 'LABEL':label, 'DESIG': desig_list, 'CPA_TIME':cpa_min}, ignore_index = True)
            # the target recording overlaps multiple hours
            else:
                end_hr = row[1]['END_TIME'].hour
                current_hr = row[1]['START_TIME'].hour + 1
                startMin = datetime.time(0,0,0)
                while current_hr != end_hr:
                    cpa_min = cpa_obj.get_cpa_min(current_hr)
                    if current_hr < 10:
                        hr2 = '0' + str(current_hr)
                    else:
                        hr2= str(current_hr)
                    extractFile = file_start+ yr + month+ day+ "_" + hr2 +file_ender
                    # this file is an entire hour, from 00:00 to 59:59
                    sliceData = sliceData.append({'MMSI':mmsi, 'FILE_NAME':extractFile, 'START_TIME':startMin, 'END_TIME':endMin, 'LABEL':label, 'DESIG': desig_list, 'CPA_TIME':cpa_min}, ignore_index = True)
                    current_hr += 1
                # fall out of while loop, now this is the last hour, start time is 00:00, end time is min of hour
                cpa_min = cpa_obj.get_cpa_min(current_hr)
                if current_hr < 10:
                    hr2 = '0' + str(current_hr)
                else:
                    hr2= str(current_hr)
                extractFile = file_start+ yr + month+ day+ "_" + hr2 +file_ender
                endMin = datetime.time(0, row[1]['END_TIME'].minute, row[1]['END_TIME'].second)
                sliceData = sliceData.append({'MMSI':mmsi, 'FILE_NAME':extractFile, 'START_TIME':startMin, 'END_TIME':endMin, 'LABEL':label, 'DESIG': desig_list, 'CPA_TIME':cpa_min}, ignore_index = True)
                
        # segments contained within the same hour
        else:
            extractFile = file_start+ yr + month+ day+ "_" + hr +file_ender
            endMin = datetime.time(0, row[1]['END_TIME'].minute, row[1]['END_TIME'].second)
            sliceData = sliceData.append({'MMSI':mmsi, 'FILE_NAME':extractFile, 'START_TIME':startMin, 'END_TIME':endMin, 'LABEL':label, 'DESIG':desig_list, 'CPA_TIME':cpa_min}, ignore_index = True)

        #######################
        # DEBUG
        #if ctr % 20 == 0:
        #    print("Completed: " + str(ctr))
        #######################
        
        # create a dict by mmsi to hold only bearing and range data,
        # data format is (time, bearing, range)
        if row[1]['LABEL'] != 'not ship':
            brg_rng_data = row[1]['BEARING_RANGE_DATA']
            # if mmsi is not in brg_rng_dict, add new dict
            if mmsi not in brg_rng_dict:
                brg_rng_dict[mmsi] = {}
            #for datum in brg_rng_data:
            #    brg_rng_dict[mmsi][datum[0]] = (datum[1], datum[2]) 
            for idx in range(0, len(brg_rng_data)-1):
                time_diff = brg_rng_data[idx+1][0] - brg_rng_data[idx][0]
                time_list = [brg_rng_data[idx][0] + datetime.timedelta(0,x) for x in range(0,time_diff.seconds)]
                # get interpolated data
                temp_brg_data = interpolate(brg_rng_data[idx][1], brg_rng_data[idx+1][1], brg_rng_data[idx][0], brg_rng_data[idx+1][0], 1)
                temp_rng_data = interpolate(brg_rng_data[idx][2], brg_rng_data[idx+1][2], brg_rng_data[idx][0], brg_rng_data[idx+1][0], 1)
                for time, bearing, tgt_range in zip(time_list, temp_brg_data, temp_rng_data):
                    brg_rng_dict[mmsi][time] = (bearing, tgt_range) 
    # write out to csv
    sliceData.to_csv(os.path.join(cwd,'tmp/labels.csv'), index=False)

    # also pickle file bearing and range data only
    with open('tmp/bearing_range_obj.pkl', 'wb') as write_file:
        pickle.dump(brg_rng_dict, write_file)

        write_file.close()

# the functions below concatanate half hour audio files into hour long files
def concat_files_multi_process(file_tuple, dir, tgt_dir):
    SR = 8000
    SR_2 = 4000
    new_front_matter = "mars_data_"

    file_a_name = file_tuple[0]
    file_b_name = file_tuple[1]
    file_c_name = file_tuple[2] 
    time_b = file_b_name.split('_')[2]
    yr = int(time_b[2:4])
    mon = int(time_b[4:6])
    day = int(time_b[6:8])
    hr = int(time_b[8:10])
    mm = int(time_b[10:12])
    sec = int(time_b[12:14])
    
    # get time from file c
    time_c = file_c_name.split('_')[2]
    mm_c = int(time_c[10:12])
    sec_c = int(time_c[12:14])

    # find offset for first file, must be in seconds
    # offset is calculated from the length of the file, 30 min 0 sec
    offset = ((29 - mm)*60 + (61 - sec))
    
    # find duration for last file, must be in seconds
    dur = ((59-mm_c)*60 + (59-sec_c)) -1
    file_a, sr = sf.read(os.path.join(dir, file_a_name), start=(offset*SR))
    file_b, sr = sf.read(os.path.join(dir, file_b_name))
    file_c, sr = sf.read(os.path.join(dir, file_c_name), stop=((dur*SR)))

    # join files by appending numpy nd arrays
    # transpose first to make concate easy
    file_a = file_a.transpose()
    file_b = file_b.transpose()
    file_c = file_c.transpose()
    temp_data = []
    for idx, channel in enumerate(file_a):
        data_to_ds = np.concatenate((channel, file_b[idx], file_c[idx]))
        temp_data.append(scipy_signal.resample(data_to_ds, int(len(data_to_ds)* SR_2/SR)))

    # must be in frames X channels format
    new_file_data = np.array(temp_data).transpose()
    # name formatting       
    if day < 10:
        day = '0' + str(day)
    else:
        day = str(day)
    if mon < 10:
        mon = '0'+str(mon)
    else: 
        mon = str(mon)
    if hr < 10:
        hr = '0' + str(hr) # add 0 to front of hour
    else:
        hr = str(hr)

    new_name = new_front_matter + str(yr) + mon + day + "_" + hr + ".wav"
    
    # write out new file in 16 bit
    sf.write(os.path.join(tgt_dir, new_name), new_file_data, samplerate=SR_2, subtype='PCM_16')
    return new_name

def create_hour_files(dir, tgt_dir):
    """
    This function will take the 30 min audio files and turn them into 1 hr.
    It will also tranform them into 16-bit from 32-bit 
    File name format is m209_oxyz_YYYYMMDDHHMMSS.wav
    #Arguments: dir: directory of files to merge
                tgt_dir: directory to put the new files in
    
    Most files do not start at the top or bottom of the hour, therefore it takes 3 files to make 1 hour file
    28min + 30min + 2min
      A       B       C   -> on the next hour, the C file becomes the A file and new B & C files are added

      A-> offset = 59:59 - time in title (ie 28:08) = offset of 1:52, no duration
      B-> no offset, no duration, whole file
      C-> no offset, duration = 59:59 - start time = ~2 min
    """

    #get all files
    max_cpus = os.cpu_count() -1
    consecutive_files = 1993
    # you might encounter a need to have a start and the end 
    files = sorted([os.path.basename(x) for x in glob.glob(dir + '*.wav')])[:consecutive_files]
    print(files[0])
    # organize into tuples for merging
    file_tuples = []
    
    for idx in range(2,len(files)-1, 2):
       file_tuples.append((files[idx-1], files[idx], files[idx+1])) 
       print(str(files[idx-1]) +" "+str(files[idx]) +" "+str(files[idx+1]))
    
    # make iterable list to pass additional args
    tgt_dir_list = [tgt_dir] * len(file_tuples)
    dir_list = [dir] * len(file_tuples)

    # now iterate through file tuples
    # perform in parallel due to time cost of writing out
    # no concurrency issues because old files are only read in, new files written out
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cpus) as executor:
        for new_file in executor.map(concat_files_multi_process, file_tuples, dir_list, tgt_dir_list):
            print("Completed: " + str(new_file))


def main():
    """
    Function to process command line arguments and create function path
    """
    str_to_bool = lambda x : True if x.lower() == 'true' else False
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    type=str,         default='.',           help="Location of data files to be divided", dest='data_dir')
    parser.add_argument('--target_dir',  type=str,         default='.',           help="Location to put output sample files",  dest='target_dir')
    parser.add_argument('--sample_rate', type=int,         default= 4000,         help="Sample rate of input audio files",     dest='sample_rate')
    parser.add_argument('--csv_file',    type=str,         default='labels.csv',  help="CSV or Excel file contatining audio files to parse", dest='csv_file')
    parser.add_argument('--duration',    type=int,         default= 30,           help="Duration of clips to cut")
    parser.add_argument('--mode',        type=str,         default='phys',        help="The mode determines which function path to take. Options are: harp, phys, from_excel, or function_test. Function_test allows for running an individual function for testing.")
    parser.add_argument('--test_fcn',    type=str,         default='main',        help="Name of function to test if mode is function_test")
    
    parser.add_argument('--pickle',      type=str_to_bool, default='False',       help="If True then the loaded file is a csv file, not csv")
 
    parser.add_argument('--cpa_label',   type=str_to_bool, default='False',       help="If Ture then cpa labels of closing or open are included, used for multi-label data")
    parser.add_argument('--multi_label', type=str_to_bool, default='False',       help="If True then multi-label data is used as input and output")
    parser.add_argument('--bearing',     type=str_to_bool, default='False',       help="If True then bearing labels are added  to samples")
    parser.add_argument('--range',       type=str_to_bool, default='False',       help="If True then range labels are added to samples")
 
    parser.add_argument('--label_type',  type=str,         default='mmsi',        help="Type of labeling system to use, either mmsi for labels based on tonnage or class for class based.")


    args = parser.parse_args()

    # set variables from command line inputs
    DATA_FOLDER = args.data_dir
    # this is the destincation folder for sample files
    TARGET_FOLDER = args.target_dir
    # sample rate of files in data_dir
    SAMPLE_RATE = args.sample_rate
    LABEL_CSV = args.csv_file
    DURATION = args.duration
    CPA_LABELS = args.cpa_label
    MULTILABEL_FLAG = args.multi_label

    # if a tmp/ folder does not exist, make it
    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')

    # create different data paths 
    # used for troubleshooting / testing individual functions
    if args.mode == 'function_test':
        if args.test_fcn == 'from_excel_file':
            from_excel_file(LABEL_CSV)
        elif args.test_fcn == 'audioClipper':
            audioClipper(LABEL_CSV, DATA_FOLDER, TARGET_FOLDER, SAMPLE_RATE, DURATION)
        elif args.test_fcn == 'create_hour_files':
            create_hour_files(DATA_FOLDER, TARGET_FOLDER)
        elif args.test_fcn == 'make_time_segments_harp':
            make_time_segments(LABEL_CSV, MULTILABEL_FLAG, args.label_type, 'harp', CPA_LABELS, args.pickle)
        elif args.test_fcn == 'make_time_segments_phys':
            make_time_segments(LABEL_CSV, MULTILABEL_FLAG, args.label_type, 'phys', CPA_LABELS, args.pickle)
        elif args.test_fcn == 'make_samples':
            make_samples_brg_rng(LABEL_CSV, DURATION, args.bearing, args.range)
        elif args.test_fcn == 'create_hour':
            create_hour_files(DATA_FOLDER, TARGET_FOLDER)
        elif args.test_fcn == 'overlap':
            overlapping_ships(LABEL_CSV)
        else:
            sys.exit("WARNING: Test Function does not exist!")
        
        sys.exit("FUNCTION TEST COMPLETE") # done, quit
    
    # used to extract data from an excel file 
    elif args.mode == 'from_excel':
        from_excel_file(LABEL_CSV)
        # right now only HARP data has an associated excel file
        make_time_segments('tmp/labels.csv', MULTILABEL_FLAG, args.label_type, 'harp', CPA_LABELS, args.pickle)
        make_samples('tmp/excel_labels.csv', DURATION)
    else:
        make_time_segments(LABEL_CSV, MULTILABEL_FLAG, args.label_type, args.mode, CPA_LABELS, args.pickle)
        if CPA_LABELS:
            make_samples_cpa('tmp/labels.csv', DURATION)
        elif args.bearing or args.range:
            make_samples_brg_rng('tmp/labels.csv', DURATION, args.bearing, args.range)
        else:
            make_samples('tmp/labels.csv', DURATION)
    
    #audioClipper('tmp/full_list_labels.csv', DATA_FOLDER, TARGET_FOLDER, SAMPLE_RATE, DURATION)
    
if __name__ == "__main__":
    main()

from utilities.time_ops import *
import pandas as pd
import glob
import os
import datetime
import numpy as np
import random


class DataHandler:

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.start_date = args.start_date
        self.end_date = args.end_date
        self.seed = args.seed
        self.shuffle_group = args.shuffle_group
        self.test_data_type = args.test_data_type

    
    def make_dataset(self):

        train_files = []
        validation_files = []
        test_files = []

        full_file_list = glob.glob(os.path.join(self.data_dir,"*include.tfrecords"))
        file_list = []

        start_date = datetime.datetime.strptime(self.start_date, '%Y%m%d %H%M%S')
        end_date = datetime.datetime.strptime(self.end_date, '%Y%m%d %H%M%S')

        for fname in full_file_list:
            file_start_date, file_end_date = file_date(fname, 'date', 'both')

            '''
            element_list = re.split('_|-|\.',fname)
            file_end_date = datetime.datetime.strptime(element_list[-3], "%Y%m%d")
            file_start_date = datetime.datetime.strptime(element_list[-5], "%Y%m%d")
            '''

            if (start_date < file_start_date and file_start_date<=end_date<=file_end_date) \
                or (file_start_date<=start_date<=file_end_date and file_end_date<end_date) \
                or (file_start_date<=start_date<=file_end_date and file_start_date<=end_date<=file_end_date) \
                or (start_date<=file_start_date and file_end_date<=end_date):
                file_list.append(fname)
            else:
                continue


            '''
            if not (start_date <= file_start_date <= end_date) and not (start_date <= file_end_date <= end_date):
                continue
            file_list.append(fname)
            '''

        # Sort files by date
        file_list = sorted(file_list, key = file_date)

        # Convert list to dataframe with date column    

        date_list = [file_date(x, 'date') for x in file_list]

        file_df = pd.DataFrame(data={"filename":file_list,"date":date_list})

        file_df['date'] = pd.to_datetime(file_df['date'])

        # Shuffle all samples splitting by day, week, or month to ensure minimal crossover of targets in train/test/val splits
        # Based on the assumption that it is unlikely the same target with the same characteristics will be present across days, weeks, or months
        random.seed(self.seed)

        if file_df.empty:
            print("There are no tfrecords for these dates")
            file_list = []
            return file_list

        if self.shuffle_group=='day':
            file_df = file_df.groupby(pd.Grouper(key='date', freq='1D'))
            groups = [df for _, df in file_df]
            random.shuffle(groups)
            file_df = pd.concat(groups).reset_index(drop=True)

        elif self.shuffle_group=='week':
            file_df = file_df.groupby(pd.Grouper(key='date', freq='1W'))
            groups = [df for _, df in file_df]
            random.shuffle(groups)
            file_df = pd.concat(groups).reset_index(drop=True)

        elif self.shuffle_group=='month':
            file_df = file_df.groupby(pd.Grouper(key='date', freq='1M'))
            groups = [df for _, df in file_df]
            random.shuffle(groups)
            file_df = pd.concat(groups).reset_index(drop=True)

        elif self.shuffle_group=='sequential':
            file_df = file_df.sort_values(by='date')

        file_list = file_df['filename'].tolist()

        # Split array at index positions indicated by last two values
        train_files, validation_files, test_files = np.split(file_list, [int(len(file_list)*0.8), int(len(file_list)*0.9)])

        train_files = train_files.tolist()
        validation_files = validation_files.tolist()
        test_files = test_files.tolist()

        # Shuffle each split individually
        random.seed(self.seed)

        random.shuffle(train_files)
        random.shuffle(validation_files)
        random.shuffle(test_files)

        if self.test_data_type=='original_split' or self.test_data_type == 'new_split':
            return train_files, validation_files, test_files
        
        elif self.test_data_type == 'data_dir':
            return file_list

        else:
            print("Please select data_dir, original_split, or new_split as the test_data_type")

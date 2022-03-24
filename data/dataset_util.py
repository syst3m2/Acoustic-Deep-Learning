"""
    Andrew Pfau
    Sonar Classifier

    This function holds tools used during the data creation process. 
    It is used to create the train, test, validation csv files for use by the dataset_processor.py program
"""


import os
import argparse
import shutil
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
import glob
import sys
import numpy as np
import pandas as pd
import random


def multiclass_data_splitter(data_path, flag, classes, limit, train_split=0.8, test_split=0.1, val_split=0.1):
    """
     This function creates 3 csv files one each for train, test, and val
     this verison is for both multi class and multi label datasets,
     filename must be in format LABEL_MMYYDD_HR.wav

     # Arguments:
        data_path: File path to where samples are stored 
        classes:   Comma seperated list of classes, these should be the first part of the filenames of the samples
        flag:      Either multilabel or multiclass
        limit:     Maximum number of files of each class to use in dataset
    """
    labels_df = pd.DataFrame(columns = ['FILE_NAME', 'LABEL'])

    filenames = []
    labels = []

    # this case is for limiting the number of samples from each class 
    if limit > 0:
        for label in classes:
            label = label.strip()
            temp_list = [f for f in glob.glob(data_path + label + '*.wav')[:limit]]   
            filenames.extend(temp_list)

    # in this case the number of samples from each class is not a concern, take all data
    else:
        filenames = [f for f in glob.glob(data_path +'*.wav')]
    # get a list of all labels for the samples in filenames
    for file_name in filenames:
        path, name = os.path.split(file_name)
        labels.append(name.split('_')[0])
    
    # for multi label, the labels are a list, create a comma sep list of strings
    if flag=='multilabel':
        temp_labels = []
        for l in labels:
            temp = l.split('-')[:-1]
            temp_labels.append(",".join(temp))
        labels = temp_labels
    
    # performs auto shuffle
    train_files, temp_files, train_labels, temp_labels = train_test_split(filenames, labels, test_size = (test_split + val_split), stratify=labels)

    # created list of training files, now write out to csv
    labels_df = pd.DataFrame(zip(train_files, train_labels), columns = ['FILE_NAME', 'LABEL'])
    labels_df.to_csv(os.path.join(data_path, 'train_labels.csv'), index=False)

    # allow for only training and validation splits of the dataset (2 splits vs 3)
    if test_split > 0.0:
        test_files, val_files, test_labels, val_labels = train_test_split(temp_files, temp_labels, test_size = 0.5, stratify=temp_labels)
        # created list of files, from temp, now write out to csv
        test_df = pd.DataFrame(zip(test_files, test_labels), columns = ['FILE_NAME', 'LABEL'])
        test_df.to_csv(os.path.join(data_path, 'test_labels.csv'), index=False)
        
        # created list of files, from temp, now write out to csv
        val_df = pd.DataFrame(zip(val_files, val_labels), columns = ['FILE_NAME', 'LABEL'])
        val_df.to_csv(os.path.join(data_path, 'val_labels.csv'), index=False)

    else:
        # created list of files, from temp, now write out to csv
        labels_df = pd.DataFrame(zip(temp_files, temp_labels), columns = ['FILE_NAME', 'LABEL'])
        labels_df.to_csv(os.path.join(data_path, 'test_labels.csv'), index=False)


def k_fold_generator(data_path, classes, limit, k):
    """
        Generate K sets of data for k-fold cross validation using sklearn StraifiedKFold
        This function can only handle multi class datasets.
        
        # Arguments:
            data_path: File path to where samples are stored 
            classes:   Comma seperated list of classes, these should be the first part of the filenames of the samples
            limit:     Maximum number of files of each class to use in dataset
            k:         Number of k-folds
        
        #Returns:
            creates k csv files called data_split_X.csv where X is the number of the split starting at 0 to k-1
    """
    
    skf = StratifiedKFold(n_splits=k)
    
    filenames = []
    labels = []
    # same logic as in multiclasss_data_splitter
    # this case is for limiting the number of samples from each class 
    if limit > 0:
        for label in classes:
            # remove any whitespaces
            label = label.strip()
            temp_list = [f for f in glob.glob(data_path + label + '_*.wav')[:limit]]   
            filenames.extend(temp_list)

    # in this case the number of samples from each class is not a convern, take all data
    else:
        filenames = [f for f in glob.glob(data_path +'*.wav')]
    
    # filenames and labels are numpy arrays to support slicing by sklearn 
    filenames_np = np.array(filenames)
    for idx, file_name in enumerate(filenames_np):
        path, name = os.path.split(file_name)
        labels.append(name.split('_')[0])

    labels_np = np.array(labels)

    idx = 0
    for _, test_idx in skf.split(filenames_np, labels_np):
        # make csv files
        # for k folds, split will return 1 fold to test and k-1 folds to train, only want to
        # save one fold each time for k csv files
        save_filenames = filenames_np[test_idx]
        save_labels = labels_np[test_idx]
        # write out to csv, shuffle before write
        labels_df = pd.DataFrame(zip(save_filenames, save_labels), columns = ['FILE_NAME', 'LABEL'])
        labels_df = shuffle(labels_df) 
        labels_df.to_csv(os.path.join(data_path, 'data_split_'+str(idx) +'.csv'), index=False)
        idx += 1

def generate_test_subset(data_path, limit):
    """
        This function generates a test_labels.csv from a subset of the data for bootstrapping
        #Arguments:
            limit: max number of files to add to csv
    """

    labels_df = pd.DataFrame(columns = ['FILE_NAME', 'LABEL'])

    # get all filenames
    filenames = [f for f in glob.glob(data_path + '*.wav')] 

    # shuffle filenames
    random.shuffle(filenames)
    labels = [] 
    for file_name in filenames:
        path, name = os.path.split(file_name)
        labels.append(name.split('_')[0])

    labels_df = pd.DataFrame(zip(filenames, labels), columns= ['FILE_NAME', 'LABEL']) 
    # save labels csv with data
    labels_df.to_csv(os.path.join(data_path, "test_labels.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir',  type=str, help="Target directory to create csv files from")
    parser.add_argument('--mode',        type=str, help="Function to use, either binary, multiclass, subset, or kFold",              default='multiclass')
    parser.add_argument('--limit',       type=int, help="Limit number of files to get",                                              default=1000)
    parser.add_argument('--k',           type=int, help="Number of folds for k fold, will create this many csv files of equal size", default=10)
    parser.add_argument('--classes',     type=str, help="Comma seperated list of classes that data is split into. For subset this is only one class")
    parser.add_argument('--csv_file',    type=str, help="CSV file of files and labels, for from_file mode")

    args = parser.parse_args()

    path = args.target_dir
    function = args.mode    

    if args.classes:
        classes = args.classes.split(',')
    else:
        sys.exit("WARNING: classes argument is not specified")
    
    if function == "multiclass" or function == "multilabel":
        multiclass_data_splitter(path, function, classes, args.limit)
    
    elif function == "subset":
        generate_test_subset(path, args.limit)
    
    elif function == "kFold":
        k_fold_generator(path, classes, args.limit, args.k)
    else:
        sys.exit("WARNING: incorrect function argument given")

if __name__ == "__main__":
    main()
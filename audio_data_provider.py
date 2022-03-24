import glob
import json
import math
import os
import random
import re
import datetime


class AudioDataProvider:
    """
    provides the audio data in the format needed to perform test, validation and training functions
    this class assumes that other pre-processing steps have been performed to generate tfrecords and
    json files which contain metadata associated with those records. those functions can be incorporated
    into this class, if necessary, at a later time
    
    Attributes
    ----------
    test_pct : number
        whole number percentage of data that should be in the test set
    vald_pct : number
        whole number percentage of data that should be in the validation set
    train_pct : number
        whole number percentage of data that should be in the training set
    json_dir : str
        the location where the tfrecord metadata is located
    test_data_split : list
        the test data split
    validation_data_split : list
        the validation data split
    train_data_split : list
        the training data split
    multi_label_class_files : dict
        the multi-label class representation of the tfrecord dataset files that were processed
    status_msg : string
        a status message used to describe the state of the splits
    reduce_samples : dict
        a dict which indicates whether a particular class should be under represented in the test/validation/train datasets

    Methods
    -------
    get_data(data_set_type=None)
        returns a dictionary with the dataset determined by the data_set_type provided (train,test,validation)
    read_tfrecords()
        locates the json files containing the tfrecord metadata, storing the class data into a dictionary
    split_datasest()
        performs the data split using the instantiation parameters provided to the class
    rebalance()
        rebalances the records so that equal distribution is applied across all classes
    describe()
        provides a description of the test, validation and training data
    
    """
    
    def __init__(self, args, store_to_file=False, test_percentage=10, validation_percentage=10, train_percentage=80, min_records_per_class=200, reduce_samples=None, include_multi_label=False, seed=42):
        """
        :param json_dir:              the path to the folder containing the json files with tfrecord dataset metadata
        :param store_to_file:         sets whether to place the data into a file (default False)
        :param test_percentage:       the percentage of records to place in the test set (default 10)
        :param validation_percentage: the percentage of records to place in the validation set (default 10)
        :param train_percentage:      the percentage of records to place in the training set (default 80)
        :param min_records_per_class: the minimum number of records to set for each class, 
                                      (-1) means the rebalance should retain the existing count, this option 
                                      should be used in conjunction with reduce_samples (default 7000)
        :param reduce_samples:        when min_records_per_class is (-1), the parameter is used to reduce a specific class 
                                      by a specified whole number percentage. must be provided as a dict
        :param include_multi_label:   identifies whether multi-label classes should be included in the split datasets (default False)
        """
        self.test_pct                = test_percentage
        self.vald_pct                = validation_percentage
        self.train_pct               = train_percentage
        self.json_dir                = args.data_dir #json_dir
        self.store_to_file           = store_to_file
        self.min_records_per_class   = min_records_per_class
        self.include_multi_label     = include_multi_label
        self.reduce_samples          = reduce_samples
        self.start_date              = datetime.datetime.strptime(args.start_date, '%Y%m%d %H%M%S')
        self.end_date                = datetime.datetime.strptime(args.end_date, '%Y%m%d %H%M%S')
        self.shuffle                 = args.data_shuffle
        self.seed                    = seed

        self.multi_label_class_files = {}
        self.status_msg              = "inited"

        #read the records and do the data split
        self.read_tfrecords()
        

    def describe(self):
        """
        Describes the current state of the test, validation and training data
        """
        print(self.status_msg)

        for _k in self.multi_label_class_files:
            if "count" in self.multi_label_class_files[_k]:
                print("Class Count [{}]: {}".format(_k,self.multi_label_class_files[_k]["count"]))

    def get_data(self, data_set_type=None):
        """
        Gets the data split as determined by the parameter supplied

        Parameters
        ----------
        data_set_type : str
            should be one of either test, train, validation or all
            when requesting "all" data, the method will return all the unique
            tfrecord files from all labels identified

        Returns
        ----------
        list 
            the list of files that are represented in the dataset
        """
        if data_set_type == 'test':
            return self.test_data_split
        elif data_set_type == 'train':
            return self.train_data_split
        elif data_set_type == 'validation':
            return self.validation_data_split
        elif data_set_type == 'all':
            return list(self.all_data)

    
    def rebalance(self):
        """
        Rebalances the datasets so that each class has the same number
        of tfrecord files represented

        this process examines the number of records found for each class,
        randomly removes files for those that contain more than self.min_records_per_class,
        and randomly repeats files for those classes that have less than self.min_records_per_class

        when self.min_records_per_class is -1, the rebalance will retain the existing file counts, but
        when used in conjuction with self.reduce_samples, will reduce the number of samples for the
        class specified
        """
        random.seed(self.seed)
        class_representation = {}
        for _k in self.multi_label_class_files:
            if _k.find("_") == -1 or self.include_multi_label:
                if "files" in self.multi_label_class_files[_k]:
                    class_representation[_k] = {
                        "count" : 0,
                        "files" : self.multi_label_class_files[_k]["files"]
                    }
                    f_count = self.multi_label_class_files[_k]["count"]
                    if self.min_records_per_class != -1:
                        if f_count < self.min_records_per_class:
                            self.expand_class(f_count,self.min_records_per_class,class_representation,_k)
                        if f_count > self.min_records_per_class:
                            for i in range(0,f_count - self.min_records_per_class):
                                rem = random.choice(class_representation[_k]["files"])
                                class_representation[_k]["files"].remove(rem)
                    else:
                        if self.reduce_samples != None and _k in self.reduce_samples:
                            nf_count = math.floor(f_count * ((100-self.reduce_samples[_k])/100))
                            if nf_count < f_count:
                                class_representation[_k]["files"] = random.choices(self.multi_label_class_files[_k]["files"],k=nf_count)
                            else:
                                self.expand_class(f_count,nf_count,class_representation,_k)
                        else:
                            class_representation[_k]["files"] = self.multi_label_class_files[_k]["files"]
                            
                    class_representation[_k]["count"] = len(class_representation[_k]["files"])
                    self.multi_label_class_files[_k]["count"] = class_representation[_k]["count"]
                    self.multi_label_class_files[_k]["files"] = class_representation[_k]["files"]

        self.split_dataset()

    def expand_class(self,f_count,min_recs,class_representation,class_key):
        tot_iterations = math.floor(min_recs/f_count)
        rem_rec_count  = min_recs % f_count
        for i in range(1,tot_iterations):
            data_padding = random.choices(self.multi_label_class_files[class_key]["files"],k=f_count)
            class_representation[class_key]["files"].extend(data_padding)
        if rem_rec_count > 0:
            data_padding = random.choices(self.multi_label_class_files[class_key]["files"],k=rem_rec_count)
            class_representation[class_key]["files"].extend(data_padding)

    
    def read_tfrecords(self):
        """
        initialization helper method which reads the json files created with the 
        tfrecords in order to create splits
        
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
        """
        
        for _file in glob.glob("{}/*.json".format(self.json_dir)):
            # Check if the date of the file is within the desired date range, if not, then discard
            
            element_list = re.split('_|-|\.',_file)
            file_end_date = datetime.datetime.strptime(element_list[-3], "%Y%m%d")
            file_start_date = datetime.datetime.strptime(element_list[-5], "%Y%m%d")
            if not (self.start_date <= file_start_date <= self.end_date) and not (self.start_date <= file_end_date <= self.end_date):
                #print(element_list[-3] + " is not in the desired date range")
                continue
            


            with open(os.path.join(self.json_dir,_file)) as tf_json:
                metadata = json.load(tf_json)
                if "MultiLabel Class Count" in metadata:
                    for key in metadata["MultiLabel Class Count"].keys():
                        # format the key from Class A,Class C => a_c, where Class C,Class A is also a_c
                        ml_labels = key.lower().replace("class","").replace(" ","").split(",")
                        sorted(ml_labels)
                        _k = "_".join(ml_labels)
                        if _k not in self.multi_label_class_files:
                            self.multi_label_class_files[_k] = { "count" : 0, "files" : [] }
                        self.multi_label_class_files[_k]["count"] = self.multi_label_class_files[_k]["count"] + 1
                        self.multi_label_class_files[_k]["files"].append(_file.replace(".json",".tfrecords"))
                        # this creates entries in the json file representing the multi-label classes 
                        # and the files which include examples containing them

        # shuffle the file lists for each of the ml classes
        #if self.shuffle:
        self.split_dataset()


        random.seed(self.seed)

        '''
        for k in self.multi_label_class_files.keys():
            self.multi_label_class_files[k]["files"] = \
                random.sample(self.multi_label_class_files[k]["files"], len(self.multi_label_class_files[k]["files"]))

        '''

        #self.test_data_split         = random.shuffle(self.test_data_split)
        #self.validation_data_split   = random.shuffle(self.validation_data_split)
        #self.train_data_split        = random.shuffle(self.train_data_split)
        #self.all_data                = random.shuffle(self.all_data)

    def split_dataset(self):
        """
        performs the data splits using the classes identified in the tfrecords

        the dictionary created from this method (which can also be optionally exported to "data_splits.json") 
        has keys for the single/multi-label classes and for the test, validation, and training data splits. 
        the single/multi-label class keys identify how many tfrecords are included for given class and the tfrecord 
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

        self.test_data_split         = []
        self.validation_data_split   = []
        self.train_data_split        = []
        self.all_data                = set()

        # Sort the full file list by date

        # Generate the test/validation/training datasets
        for k in self.multi_label_class_files.keys():
            self.all_data.update(self.multi_label_class_files[k]["files"])
            if k.find("_") == -1 or self.include_multi_label:
                ml_ds = self.multi_label_class_files[k]

                test_count = (self.test_pct/100) * ml_ds["count"]
                test_count = math.floor(test_count)
                _test_list = ml_ds["files"][0:test_count-1]
                self.test_data_split.extend(_test_list)
                
                vald_count = (self.vald_pct/100) * ml_ds["count"]
                vald_count = math.floor(vald_count)
                _vald_list = ml_ds["files"][test_count:test_count+vald_count-1]
                self.validation_data_split.extend(_vald_list)

                train_count = (self.train_pct/100) * ml_ds["count"]
                train_count = math.floor(train_count)
                _train_list = ml_ds["files"][test_count+vald_count:]
                self.train_data_split.extend(_train_list)

        ml_test_vald_train = {
            "test_set"       : self.test_data_split,
            "validation_set" : self.validation_data_split,
            "train_set"      : self.train_data_split
        }

        ml_class_files = {**self.multi_label_class_files, **ml_test_vald_train}

        self.status_msg = \
            "\n###################\ntest_set size: {}\nvalidation_set size: {}\ntrain_set size: {}\n###################\n".format(\
                    len(self.test_data_split),len(self.validation_data_split),len(self.train_data_split))

        if self.store_to_file:
            data_splits_json = json.dumps(ml_class_files)
            with open("data_splits.json", "w") as outfile:
                json.dump(data_splits_json, outfile)

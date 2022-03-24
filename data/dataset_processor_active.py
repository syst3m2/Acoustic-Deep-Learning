"""
    Andrew Pfau
    Sonar Classifier

    This is the data input pipeline code. It processes audio samples into spectrograms according to the given parameters

"""

import os
import numpy as np
import tensorflow as tf
import numpy as np
from pathlib import Path
import glob
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer

class AudioDataset(object):

    def __init__(self,
                    win_size,
                    overlap,
                    sample_points,
                    sample_rate,
                    mel_bins,
                    data_filepath=None,
                    batch_size=2,
                    epochs=2,
                    buffer_size=10,
                    file_format='wav',
                    augment=False,
                    repeat = False,
                    modelType = "multi-class",
                    dur = 30,
                    mode='train'):

                    '''
                    collect all parameters and make a dataset:

                    Args:
                    train_filepath: pathlib object to training files
                    BATCH_SIZE: default is 2, batch size used in the training
                    EPOCH = 2: default is 2, how many times to iterate through the data
                    dont put commas after self.XX  , it will make tuples!
                    '''
                    self.data_filepath=data_filepath
                    self.batch_size=batch_size
                    self.epochs=epochs
                    self.buffer_size=buffer_size
                    self.file_format=file_format
                    self.augment = augment
                    self.duration = dur
                    self.mode = mode
                    self.model_type = modelType
                    self.repeat = repeat
                    self.filepaths = []
                    self.labels = []

                    # spectrogram params
                    # win size comes in as seconds-> convert to number of samples by multipling by sample rate
                    # win size must produce an integer
                    # overlap comes in a % -> multiply by win size to get sample overlap
                    
                    self.SAMPLE_RATE = sample_rate
                    # These calcuations only need to be made once for all spectrograms
                    # used to convert time hparam to number of sample points for tf.signal.stft
                    # 0.001 converts msecs to seconds
                    self.WIN_SIZE = (int)((win_size * .001) * self.SAMPLE_RATE )
                    self.OVERLAP = overlap
                    # converts a overlap percent into number of sample points
                    self.STEP = (int)((self.OVERLAP /100) * self.WIN_SIZE)
                    self.SAMPLE_POINTS = sample_points
                    # mel bins is only used in MFCCs
                    self.MEL_BINS = mel_bins
    """
        Getter functions to return true labels array and filepaths array
        These functions are used during prediction with the saved model
    """
    def get_true_labels(self):
        return self.labels

    def get_filepaths(self):
        return self.filepaths


    def make_dataset(self, csv_file, generator_type):
        """
        create the actual dataset
        The dataset consists of spectrogram images and a metadata file with the labels
        This is required to support multiple labels per image
        
        #Arguments
            csv_file: file located in data_filepath in object constructor the contains a list of files and associated labels
            generator: either 'stft' or 'mfcc' for which output to transform audio files into

        #Returns
            dataset: tensorflow dataset object of spectrograms and lables
            size: size of dataset, gets around some tensorflow eager execution to get know dataset size from length in csv file
        """
        # open and read the csv file at train_filepath, assumed to be in the format FILE_NAME, LABELS
        # This reads in the csv and then turns the files pointed to in FILE_NAMES into spectrograms
        
        target_file = os.path.join(self.data_filepath,csv_file)
        data = np.loadtxt(target_file, dtype = 'str', delimiter=',', skiprows=1)
        # for multi label cases we have to add 2 or more columns together to get the able set
        last_col = np.size(data, 1)
        labels = data[:,1:last_col] 
        labels = [[ele for ele in row if ele != ''] for row in labels]
        size = len(data.T[0])
        # get the count of each class
        v, counts = np.unique(labels, return_counts=True)
        ordered_counts = sorted(zip(v,counts), key=lambda x:x[0])
        counts = [ c for v,c in ordered_counts] 
        
        print("Number of file paths in " +csv_file + " " +str(size)) 
        # make the list of files from the csv a tensor of filepaths 
        filepaths = tf.data.Dataset.from_tensor_slices(data.T[0])

        # convert labels to one hot encoding
        label_vector = []
        if self.model_type == "multi_label":
            label_vector = self.multi_label_one_hot_encoder(labels)
        # format of file names in classification, bearing, range
        elif self.model_type == "regression_bearing":
            label_vector = [float(row[1]) for row in labels]
        
        elif self.model_type == "regression_range":
            label_vector = [float(row[2].replace('"', '')) for row in labels]
        
        elif self.model_type == "regression_bearing_range":
            label_vector = [(float(row[1]), float(row[2].replace('"',''))) for row in labels]
        
        else: # multi_class case
            label_vector = self.one_hot_encoder([row[0].replace('"','') for row in labels])

        # save labels and filepaths for inference, if the mode is regression, directly save the label_vector since
        # it is just numbers, otherwise, save the true labels
        if(self.mode == 'saved' or self.mode =='active'):
            if 'regression' in self.model_type:
                self.labels = label_vector
            else:
                self.labels = labels    
            
            self.filepaths = data.T[0]        

        #convert labels into dataset to merge with spectrograms
        label_tensor = tf.data.Dataset.from_tensor_slices(label_vector)

        # merge filepaths and labels
        dataset = tf.data.Dataset.zip((filepaths, label_tensor))
      
        # call the process_path function to create the spectrograms
        # lambda just splits out the filepath and label for the function call
        # returns a spectrogram array and label pair

        if generator_type == 'stft':
            dataset = dataset.map(lambda x, y: self.generate_stft(x, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        elif generator_type =='mfcc':
            dataset = dataset.map(lambda x, y: self.generate_mfcc(x, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        elif generator_type =='mfcc_multi_channel':
            dataset = dataset.map(lambda x, y: self.generate_multi_channel_mfcc(x, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        elif generator_type =='stft_multi_channel':
            dataset = dataset.map(lambda x, y: self.generate_multi_channel_stft(x, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # data augmentation
        if self.augment:
            dataset = dataset.concatenate(dataset.map(self.augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE))
            size *= 2

        dataset = dataset.cache() 
        # shuffle dataset, only when training
        # buffer is the number of total items in the dataset that tf will
        # randomly choose from, refilling after every choice
        if('train' in csv_file):
            dataset = dataset.shuffle(self.buffer_size)

        # epoch repeat logic
        dataset = dataset.repeat(self.epochs)

        # batching logic
        dataset = dataset.batch(self.batch_size)

        # prefetch for speed
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return [dataset,size, counts] 

    def generate_stft(self, filepath, label):
        """
        This function processes the filepaths read-in from the csv file into STFTs
        """
        audio_data = tf.io.read_file(filepath)
        audio, _ = tf.audio.decode_wav(audio_data, desired_channels=1)
        
        # tf.squeeze(audio) to change shape to just samples, removes number of channels
        audio_squeeze = tf.reshape(tf.squeeze(audio), [1,-1])
        stfts = tf.signal.stft(audio_squeeze, frame_length=self.WIN_SIZE, frame_step=self.STEP, fft_length=self.SAMPLE_POINTS, window_fn=tf.signal.hann_window, pad_end=True )
        spectrograms = tf.abs(stfts)
        
        # built in tf.math.log is log base 2, need log base 10
        spectrograms = (tf.math.log(spectrograms) / tf.math.log(tf.constant(10, dtype=spectrograms.dtype)))

        # calculate the time axis for conversion
        time_space = (int)(self.duration * self.SAMPLE_RATE) // self.STEP
        # stft function returns channels, time, freq, need to convert to time, freq, channels for CNNs
        final_spec =  tf.reshape(tf.squeeze(spectrograms), [time_space, (self.SAMPLE_POINTS // 2) + 1, 1])
        if self.repeat:
            final_spec = tf.repeat(final_spec, repeats=3, axis=2)

        return final_spec, label
    
    def generate_mfcc(self, filepath, label):
        """
            This function processes the filepaths read-in from the csv file into mel log STFTs
        """        
        # MFCC Parameters
        # lower bound frequency in hz, selected to not include lower frequency DC signal
        LOWER_BOUND = 10.0
        # upper bound frequency in hz, selected to be max possible frequency
        UPPER_BOUND = 2000.0

        audio_data = tf.io.read_file(filepath)
        audio, _ = tf.audio.decode_wav(audio_data, desired_channels=1)
        
        # tf.squeeze(audio) to change shape to just samples, removes number of channels
        audio_squeeze = tf.reshape(tf.squeeze(audio), [1,-1])
        stfts = tf.signal.stft(audio_squeeze, frame_length=self.WIN_SIZE, frame_step=self.STEP, fft_length=self.SAMPLE_POINTS, window_fn=tf.signal.hann_window, pad_end=True )
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = (self.SAMPLE_POINTS //2) +1

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.MEL_BINS, num_spectrogram_bins, self.SAMPLE_RATE, LOWER_BOUND, UPPER_BOUND)
        mel_spectrograms = tf.tensordot( spectrograms, linear_to_mel_weight_matrix, 1)

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        # Compute MFCCs from log_mel_spectrograms
        #mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
        # calculate the time axis for conversion
        time_space = (int)(self.duration * self.SAMPLE_RATE) // self.STEP
        # mfcc function returns channels, time, freq, need to convert to time, freq, channels for CNNs
        final_mfcc =  tf.reshape(tf.squeeze(log_mel_spectrograms), [time_space, self.MEL_BINS, 1])
        
        if self.repeat:
            final_mfcc = tf.repeat(final_mfcc, repeats=3, axis=2)

        return final_mfcc, label

    def generate_multi_channel_mfcc(self, filepath, label):
        """
            This function processes the filepaths read-in from the csv file into mel log STFTs
            This function exists only to process multi-channel files 
        """        
        # MFCC Parameters
        # lower bound frequency in hz, selected to not include lower frequency DC signal
        LOWER_BOUND = 10.0
        # upper bound frequency in hz, selected to be max possible frequency
        UPPER_BOUND = 2000.0
        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = (self.SAMPLE_POINTS //2) +1

        audio_data = tf.io.read_file(filepath)
        audio, _ = tf.audio.decode_wav(audio_data, desired_channels=4)
        channels = tf.split(audio, num_or_size_splits=4, axis=1) 
        # tf.squeeze(audio) to change shape to just samples, removes number of channels
        all_channels = []
        for ch in channels:
            audio_squeeze = tf.reshape(tf.squeeze(ch), [1,-1])
            stfts = tf.signal.stft(audio_squeeze, frame_length=self.WIN_SIZE, frame_step=self.STEP, fft_length=self.SAMPLE_POINTS, window_fn=tf.signal.hann_window, pad_end=True )
            spectrograms = tf.abs(stfts)

            linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.MEL_BINS, num_spectrogram_bins, self.SAMPLE_RATE, LOWER_BOUND, UPPER_BOUND)
            mel_spectrograms = tf.tensordot( spectrograms, linear_to_mel_weight_matrix, 1)

            # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
            log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
            all_channels.append(tf.squeeze(log_mel_spectrograms))
        # mfcc function returns channels, time, freq, need to convert to time, freq, channels for CNNs
        final_mfcc =  tf.stack([all_channels[0], all_channels[1], all_channels[2], all_channels[3]], axis=2)

        return final_mfcc, label
    
    def generate_multi_channel_stft(self, filepath, label):
        """
            This function processes the filepaths read-in from the csv file into log STFTs
            This function exists only to process multi-channel files 
        """        
        audio_data = tf.io.read_file(filepath)
        audio, _ = tf.audio.decode_wav(audio_data, desired_channels=4)
        
        channels = tf.split(audio, num_or_size_splits=4, axis=1) 
        
        all_channels = []
        for ch in channels:  # generate spectorgram for each channel independently and put back together at the end.
            audio_squeeze = tf.reshape(tf.squeeze(ch), [1,-1])
            stfts = tf.signal.stft(audio_squeeze, frame_length=self.WIN_SIZE, frame_step=self.STEP, fft_length=self.SAMPLE_POINTS, window_fn=tf.signal.hann_window, pad_end=True )
            spectrograms = tf.abs(stfts)

            # built in tf.math.log is log base 2, need log base 10
            log_spectrograms = (tf.math.log(spectrograms) / tf.math.log(tf.constant(10, dtype=spectrograms.dtype)))

            all_channels.append(tf.squeeze(log_spectrograms))
        
        # function returns channels, time, freq, need to convert to time, freq, channels for CNNs
        final_stft =  tf.stack([all_channels[0], all_channels[1], all_channels[2], all_channels[3]], axis=2)

        return final_stft, label

    def multi_label_one_hot_encoder(self, labels):
        """ 
            This function transforms the list of labels into a one hot encoded vector
            The list is the list of all labels in the dataset
            Uses sklearn encoders. For the multilabel case

            Returns: 2D list in one hot form, or for multi-label case, multi-hot 
        """
        onehot_encoder = MultiLabelBinarizer()
        onehot_vector = onehot_encoder.fit_transform(labels)

        return onehot_vector

    def one_hot_encoder(self, labels):
        """ 
            This function transforms the list of labels into a one hot encoded vector
            The list is the list of all labels in the dataset
            Uses sklearn encoders

            Returns: 2D list in one hot form, or for multi-label case, multi-hot 
        """
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder()
        
        # convert string categories to numbers
        integer_encoded = label_encoder.fit_transform(labels)
        # Transpose integer encoding
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        # convert to one_hot_vector
        onehot_vector = onehot_encoder.fit_transform(integer_encoded).toarray()
        return onehot_vector

    def augmentation(self, image, label):
        image = tf.reverse(image, [1])
        return image, label

def test_dataset():
    """
    Function to test the speed of the dataset under varying conditions
    """
    bs = 4
    ep = 4
    # for windows file path 
    data_folder = "../clips/"
    ds = AudioDataset(
        win_size=500,
        overlap=0.5,
        sample_points=1024,
        sample_rate = 4000,
        mel_bins=20,
        data_filepath =data_folder,
        mode='saved',
        batch_size = bs,
        epochs = ep,
        modelType="regression")
    
    test_set, size, labels = ds.make_dataset('train_labels.csv', 'stft')
    print("Dataset Built") 
    #print(len(test_set)) 
    for test_output,label in test_set.take(5):
        # self.dataset=self._make_dataset() 
        print("Specgram shape: ", test_output.numpy().shape)
        print("Label: ", label.numpy())
    true_l = ds.get_true_labels()
    label_encode = MultiLabelBinarizer()
    true_labels = label_encode.fit_transform(true_l)
    print(true_labels)
    
    
    print("Dataset test completed SAT")

if __name__ == "__main__":
    test_dataset()

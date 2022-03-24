'''
    Andrew Pfau
    Thesis work
    Script to run new training model. Called from main when mode is set to 'train'
    Builds and trains a new model. See main.py for input arguments and flags.
'''

# library imports
from numpy.core.numeric import full
import tensorflow as tf
import glob
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score
from tensorboard.plugins.hparams import api as hp
import tensorflow_probability as tfp
import random
# imports from other files in project
import data.dataset_processor as data_processor
from models.model import build_model
import models.model_callbacks as callback_gen
from models.model_optimizers import get_optimizer, get_loss
from utilities.data_vis_tools.spectrogram_generator import generate_from_model
from audio_data_provider import *
from dataReader import *
import json
import math
import pandas as pd
from utilities.time_ops import *
from data_manager import *

def main(args):
    # print version of the framework and GPUs
    print("Tensorflow Version: " + tf.__version__ )
    print("GPUs available: " + str(len(tf.config.list_physical_devices('GPU'))))
    print("CPUs available: " + str(os.cpu_count()))

    # grab number of gpus automatically
    gpus = len(tf.config.list_physical_devices('GPU')) 
    batch_size = args.batch_size #* gpus
    eval_metrics = args.eval_metrics
    activation_fcn = "" 
    # create dataset
    channels = args.channels
    repeat = False
    multi_label = False
    if args.model in ['vgg', 'mobilenet', 'inception']:
        channels = 3
        repeat = True
    if args.model_type=='multi_label':
        multi_label = True
    
    
    # based on input type and number of channels, determine what generator type to use
    model_input_generator = 'stft'
    if args.model_input == 'stft' and args.channels == 4:
        model_input_generator = 'stft_multi_channel'
    elif args.model_input == 'mfcc' and args.channels == 4:
        model_input_generator = 'mfcc_multi_channel'
    elif args.model_input == 'mfcc' and args.channels == 1:
        model_input_generator = 'mfcc'

    # create individaul train, and validate datasets    
    # also returns the size of the datasets

    
    #Using audio data provider class
    #Use if you don't care about shuffling files between train, test, val datasets


    if args.dataset=='tfrecord':
        if args.test_data_type == 'original_split':
            mldata = AudioDataProvider(args)
            mldata.describe()
            test_files = mldata.get_data("test")
            validation_files  = mldata.get_data("validation")
            train_files = mldata.get_data("train")

        elif args.test_data_type == 'new_split':
            data_manager = DataHandler(args)
            train_files, validation_files, test_files = data_manager.make_dataset()


        # Get files and split sequentially
        '''
        train_files = []
        validation_files = []
        test_files = []

        full_file_list = glob.glob(os.path.join(args.data_dir,"*include.tfrecords"))
        file_list = []

        start_date = datetime.datetime.strptime(args.start_date, '%Y%m%d')
        end_date = datetime.datetime.strptime(args.end_date, '%Y%m%d')

        for fname in full_file_list:
            file_start_date, file_end_date = file_date(fname, 'date', 'both')

            if (start_date < file_start_date and file_start_date<=end_date<=file_end_date) or (file_start_date<=start_date<=file_end_date and file_end_date<end_date) or (file_start_date<=start_date<=file_end_date and file_start_date<=end_date<=file_end_date) or (start_date<=file_start_date and file_end_date<=end_date):
                file_list.append(fname)
            else:
                continue

        # Sort files by date
        file_list = sorted(file_list, key = file_date)

        # Split array at index positions indicated by last two values
        train_files, validation_files, test_files = np.split(file_list, [int(len(file_list)*0.8), int(len(file_list)*0.9)])

        train_files = train_files.tolist()
        validation_files = validation_files.tolist()
        test_files = test_files.tolist()

        # Shuffle each list
        random.seed(42)

        random.shuffle(train_files)
        random.shuffle(validation_files)
        random.shuffle(test_files)
        '''


        # Create tensorflow dataset
        print("Creating Dataset")
        train_data, train_metadata = get_audio_dataset(args, train_files, batch_size, args.data_type, 'features')
        test_dataset, test_metadata = get_audio_dataset(args, test_files, batch_size, args.data_type, 'features')
        validate_data, val_metadata = get_audio_dataset(args, validation_files, batch_size, args.data_type, 'features')

        print("\n###################\nTrain Dataset Metadata\n###################\n")
        print(train_metadata)
        print("\n###################\nTest Dataset Metadata\n###################\n")
        print(test_metadata)
        print("\n###################\nVal Dataset Metadata\n###################\n")
        print(val_metadata)

        train_data = train_data.shuffle(25000)
        train_data = train_data.repeat(args.num_epochs)
        validate_data = validate_data.repeat(args.num_epochs)
        

        # prefetch for speed
        validate_data = validate_data.batch(batch_size,drop_remainder=True)
        train_data = train_data.batch(batch_size,drop_remainder=True)
        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
        validate_data = validate_data.prefetch(tf.data.experimental.AUTOTUNE)
    

    elif args.dataset == 'wav':
        data_proc = data_processor.AudioDataset(win_size=args.win_size, overlap=args.overlap, sample_points=args.sample_pts, mel_bins=args.mel_bins ,buffer_size=(args.batch_size * 5), 
                                                sample_rate=args.sample_rate, data_filepath=args.data_dir, epochs=args.num_epochs, batch_size=args.batch_size, augment=args.data_aug, repeat=repeat,
                                                dur=args.duration, modelType=multi_label)

        train_data, ds_size, l_count = data_proc.make_dataset('train_labels.csv', model_input_generator)
        validate_data, val_size, _ = data_proc.make_dataset('val_labels.csv', model_input_generator)

    # calculate input shape parameters
    step = (args.overlap/100 )* ((args.win_size * 0.001) * args.sample_rate)
    time_axis = (int)((args.duration * args.sample_rate) // step)
    # stft frequency axis is different than MFCC, time axises are the same
    if args.model_input == 'stft':
        freq_axis = (args.sample_pts//2) +1
    else:
        freq_axis = args.mel_bins 
    input_shape = (time_axis ,freq_axis, channels)

    # set loss and final activation functions based on model type
    if args.model_type == "multi_class":
        loss_fcn = get_loss("crossentropy", args.label_smooth)
        activation_fcn = "softmax"
    elif args.model_type == "multi_label":
        loss_fcn = get_loss("binarycrossentropy", args.label_smooth)
        activation_fcn = "sigmoid"
    elif "regression" in args.model_type:
        loss_fcn = get_loss("mse", args.label_smooth)
        activation_fcn = "linear"
        eval_metrics = ["mse", "mae", "mape"] 
    else:
        sys.exit("Specificed model type not allowed")
    
    # save spectrograms
    # this aids in verifying the data pipeline is correct and that labels are correctly matched to
    # spectrograms, also helps visualize dataset
    # see spectrogram_generator.py for more
    if args.print_specgram: 
        print("Generating Spectrograms")
        for spec_batch, label_batch in train_data.take(1):
            generate_from_model(args, spec_batch, label_batch,[],[], time_axis, freq_axis)
        print("Spectrograms saved")

    if args.model == 'cnn_model_hparam':
        hparam_tuning(args, input_shape, activation_fcn, loss_fcn, (ds_size // batch_size), (val_size // batch_size), train_data, validate_data, data_proc)
        exit()

    # multi GPU
    if gpus > 1:
        print("There are multiple GPUs")
        print(gpus)
        # need to adjust batch size since each GPU gets batch size / num gpus data
        # this is necessary for calculating step size later as each gpu gets args.batch_size / num gpus samples per batch
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            #build model
            model = build_model(args, input_shape, activation_fcn,0)

            # compile model
            model.compile(optimizer = get_optimizer(args), 
                            loss= loss_fcn,
                            metrics = [eval_metrics])
            
            if args.mode == 'cont-train' or args.mode == 'active-cont-train':
                model.load_weights(args.saved_model_path)
                model.trainable = True

    else: # single gpu
        #build model
        model = build_model(args, input_shape, activation_fcn,0)

        # compile model
        model.compile(optimizer = get_optimizer(args), 
                        loss = loss_fcn,
                        metrics = [eval_metrics])

        if args.mode == 'cont-train' or args.mode == 'active-cont-train':           
            model.load_weights(args.saved_model_path)
            model.trainable = True

    # show model summary
    print(model.summary())
    
    # calculate step sizes
    step_size = train_metadata["Examples Count"] // batch_size
    val_step_size = val_metadata["Examples Count"] // batch_size


    if args.mode == 'train' or args.mode == 'cont-train':

        # This code does not work with new pipeline, need to fix l_count variable and take from metadata
        if args.class_weight:
            # use weights per class in model.fit, first calculate weights
            class_weights = {}
            total = np.sum(l_count)
            
            for idx, count in enumerate(l_count):
                class_weights[idx] = ((1/count) * (total)) / 2.0
            print("WEIGHTS: " + str(class_weights))
            
            # fit the model
            model.fit(train_data, epochs=args.num_epochs, validation_data=validate_data,
                        callbacks=callback_gen.make_callbacks(args), class_weight=class_weights)
        
        else:
        
            model.fit(train_data, epochs=args.num_epochs, validation_data=validate_data, steps_per_epoch = step_size, validation_steps=val_step_size,
                        callbacks=callback_gen.make_callbacks(args))

    # Still in development, not ready yet
    elif args.mode == 'active-train' or args.mode == 'active-cont-train':
        learning_loop(args,model,args.active_loops,64)



# Use base trained model first
# Try one using fresh, untrained model
# Define total time period
# Define time step

def learning_loop(args,model,numActiveLoops,queryNum):



    result = getBNNAcc(model,None,None)
    results.append(result)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.checkpoint_dir,'bestActive.h5'),
                                    monitor='val_accuracy',
                                    verbose=1,
                                    mode='auto',
                                    save_best_only=True,
                                    save_freq='epoch')

    callbacks_list = [checkpoint]
    
    #print('model started at '+ str(result)) 
    for x in range(numActiveLoops):   
        query_idx = queryMethods[args.queryMethod](model,queryNum,pool_data)
        
        print('Starting Active Learning Loop '+ str(x))
        step_size = ds_size // args.batch_size
        val_step_size = val_size // args.batch_size
        

        #train
        model.fit(x=train_data,epochs=args.num_epochs,validation_data=validate_data,steps_per_epoch = step_size, validation_steps=val_step_size,callbacks=callbacks_list)

        #get and save results
        
        model = tf.keras.models.load_model(os.path.join(args.checkpoint_dir,'bestActive.h5'))
        print('loaded best model')

        result = getBNNAcc(model,None,None)
        results.append(result)
        print('F1 = '+str(result))
        current_loop+=1
        save()
        
    return None



def hparam_tuning(args, input_shape, activation_fcn, loss_fcn, step_size, val_step_size, train_data, validate_data, data_proc):
    """
    This function is for hparam tuning 
    """
    test_data, _, _ = data_proc.make_dataset('test_labels.csv', args.model_input)
    #HPARAM tuning for dev_model only
    HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([3,5]))
    HP_NUM_FILTERS = hp.HParam('num_filters', hp.Discrete([24, 32, 48]))
    HP_KERNEL_RATIO = hp.HParam('kernel_ratio', hp.Discrete([1, 2, 3, 4]))
    HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([5, 10]))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.15, 0.2, 0.25]))
    HP_DENSE_LAYERS = hp.HParam('dense_layer', hp.Discrete([24, 12]))
    METRIC = 'accuracy'

    with tf.summary.create_file_writer(args.checkpoint_dir + '/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_LAYERS, HP_NUM_FILTERS],
             metrics=[hp.Metric(METRIC, display_name='Accuracy')],
        )
    
    idx = 0 
    for layer in HP_NUM_LAYERS.domain.values:
        for filters in HP_NUM_FILTERS.domain.values:
            hparams = {HP_NUM_LAYERS: layer, HP_NUM_FILTERS: filters}

            tmp = {h.name: hparams[h] for h in hparams}
            print(tmp)
            with tf.summary.create_file_writer(args.checkpoint_dir + '/hparam_tuning/run-' + str(idx)).as_default():
                hp.hparams(hparams)
                #build model, must be rebuilt each time 
                model = build_model(args, input_shape, activation_fcn, tmp)

                # compile model
                model.compile(optimizer = get_optimizer(args), 
                                loss = loss_fcn,
                                metrics = [args.eval_metrics])
                # show model summary
                print(model.summary())

                
                # fit the model
                model.fit(train_data, epochs=args.num_epochs, validation_data=validate_data, steps_per_epoch = step_size, validation_steps=val_step_size,
                            callbacks=callback_gen.make_callbacks(args))
                _, accuracy = model.evaluate(test_data)
                tf.summary.scalar(METRIC, accuracy, step=1)
                idx += 1



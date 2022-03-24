'''
    Andrew Pfau
    Thesis work
    Script to run new training model. Called from main when mode is set to 'train'
    Builds and trains a new model. See main.py for input arguments and flags.

    This version is used specifically for k-fold cross validation training.
    The mode of 'train-k-fold' must be specified for main.py arguments
'''

# library imports
import tensorflow as tf
import glob
import numpy as np
import os
from sklearn.metrics import accuracy_score
import shutil

# imports from other files in project
import data.dataset_processor as data_processor
from models.model import build_model
import models.model_callbacks as callback_gen
from models.model_optimizers import get_optimizer, get_loss
from utilities.data_vis_tools.spectrogram_generator import generate_from_model


def main(args):
    # print version of the framework and GPUs
    print("Tensorflow Version: " + tf.__version__ )
    print("GPUs available: " + str(len(tf.config.list_physical_devices('GPU'))))
    print("CPUs available: " + str(os.cpu_count()))

    batch_size = args.batch_size

    # create dataset
    data_proc = data_processor.AudioDataset(win_size=args.win_size, overlap=args.overlap, sample_points=args.sample_pts, mel_bins=args.mel_bins ,buffer_size=(args.batch_size * 5), 
                                            sample_rate=args.sample_rate, data_filepath=args.data_dir, epochs=args.num_epochs, batch_size=args.batch_size)
    
    # list to hold k datasets
    datasets = []
    for x in range(args.k):
        dataset, ds_size = data_proc.make_dataset('data_split_' +str(x)+'.csv', args.model_input)
        datasets.append(dataset)

    # calculate input shape parameters
    step = (args.overlap/100 )* ((args.win_size * 0.001) * args.sample_rate)
    time_axis = (int)((30 * args.sample_rate) // step)
    # stft frequency axis is different than MFCC, time axises are the same
    if args.model_input == 'stft':
        freq_axis = (args.sample_pts//2) +1
    else:
        freq_axis = args.mel_bins 
        time_axis += 1

    input_shape = (time_axis ,freq_axis, 1)

    # set loss and final activation functions based on model type
    if args.model_type == "multi_class":
        loss_fcn = get_loss("crossentropy", args.label_smooth)
        activation_fcn = "softmax"
    elif args.model_type == "multi_label":
        loss_fcn = get_loss("binarycrossentropy", args.label_smooth)
        activation_fcn = "sigmoid"
    else:
        print("Specificed model type not allowed")
        exit()

    # multi GPU
    if args.gpus > 1:
        # need to adjust batch size since each GPU gets batch size / num gpus data
        batch_size = args.batch_size / args.gpus 
        
        mirrored_stratgey = tf.distribute.MirroredStrategy()
        with mirrored_stratgey.scope():
            #build model
            model = build_model(args, input_shape)

            # compile model
            model.compile(optimizer = get_optimizer(args), 
                            loss= loss_fcn,
                            metrics = [args.eval_metrics])
    # single GPU
    else:
        #build model
        model = build_model(args, input_shape, activation_fcn)

        # compile model
        model.compile(optimizer = get_optimizer(args), 
                        loss = loss_fcn,
                        metrics = [args.eval_metrics])
    
    # show model summary
    print(model.summary())
    
    # calculate step sizes
    step_size = (ds_size* (args.k-2)) // batch_size
    val_step_size = ds_size // batch_size
    
    # repeat fit k times with a different validation set each time
    # idx will always indicate the slice being held out for testing later, the test results are averaged
    
    val_idx = 1
    for test_idx in range(args.k):
        #concate datasets together with concatonate operator
        # do this for all datasets except the validate and test dataset
        train_data = None
        for x in range(args.k):
            # only add data if it is not the test or validation set
            if (x != test_idx and x != val_idx):
                if train_data == None:
                    print("Adding set to none:" + str(x) + ' ' + str(test_idx))
                    train_data = datasets[x]
                else:
                    print("Adding set:" + str(x))
                    train_data = train_data.concatenate(datasets[x])
        
        model.fit(train_data, epochs=args.num_epochs, validation_data=datasets[val_idx], steps_per_epoch = step_size, validation_steps=val_step_size,
                 callbacks=callback_gen.make_callbacks(args))

        # save the checkpoint file, only the best checkpoint file is saved as 'checkpoint-kfold.h5'
        shutil.move(os.path.join(args.checkpoint_dir, 'checkpoint-kfold.h5'), os.path.join(args.checkpoint_dir, 'checkpoint_' +str(test_idx)+ '.h5'))

        # next cycle
        if (val_idx) +1 <= (args.k-1):
            val_idx += 1
        else:
            # reset to 0
            val_idx = 0
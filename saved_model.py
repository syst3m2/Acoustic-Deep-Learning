'''
    Andrew Pfau   Thesis work
    This script loads saved model parameters and resumes model
    operations, either training or predicting from the saved file.
'''


import data.dataset_processor as data_processor
import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import pickle
# imports for plotting
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss, multilabel_confusion_matrix, roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from models.model_optimizers import get_optimizer, get_loss
from models.model import build_model
import copy 
from audio_data_provider import *
from dataReader import *
from utilities.data_vis_tools.spectrogram_generator import generate_from_model
from utilities.time_ops import *
from plots import *
from data_manager import *

bnn = True

def BNN_predict(num_classes,to_test):
    pred_vi=np.zeros((len(to_test),num_classes))
    pred_max_p_vi=np.zeros((len(to_test)))
    pred_std_vi=np.zeros((len(to_test)))
    entropy_vi = np.zeros((len(to_test)))
    norm_entropy_vi =  np.zeros((len(to_test)))
    epistemic = np.zeros((len(to_test)))
    aleatoric = np.zeros((len(to_test)))
    var=  np.zeros((len(to_test)))
    for i in range(0,len(to_test)):
        preds = to_test[i]
        pred_vi[i]=np.mean(preds,axis=0)#mean over n runs of every proba class
        pred_max_p_vi[i]=np.argmax(np.mean(preds,axis=0))#mean over n runs of every proba class
        pred_std_vi[i]= np.sqrt(np.sum(np.var(preds, axis=0)))
        var[i] =  np.sum(np.var(preds, axis=0))
        entropy_vi[i] = -np.sum( pred_vi[i] * np.log2(pred_vi[i] + 1E-14)) #Numerical Stability
        epistemic[i] = np.sum(np.mean(preds**2, axis=0) - np.mean(preds, axis=0)**2)
        aleatoric[i] = np.sum(np.mean(preds*(1-preds), axis=0))
        norm_entropy_vi[i] = entropy_vi[i]/np.log2(2^num_classes)
    pred_vi_mean_max_p=np.array([pred_vi[i][np.argmax(pred_vi[i])] for i in range(0,len(pred_vi))])
    nll_vi=-np.log(pred_vi_mean_max_p)


    return pred_vi, pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi, var, norm_entropy_vi, epistemic, aleatoric

def classification_metrics_mc(true_labels, pred_labels_probs):
    pred_labels = pred_labels_probs.argmax(-1)
    accuracy = accuracy_score(true_labels, pred_labels)
    print("Accuracy: %.3f" % accuracy)

    print(classification_report(true_labels, pred_labels, target_names=['a', 'b','c','d','e']))
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print(str(conf_matrix))

    lr_auc = roc_auc_score(true_labels, pred_labels_probs, multi_class='ovr')
    print(' aucROC=%.3f' % (lr_auc))

    hloss = hamming_loss(true_labels, pred_labels)
    print("Hamming loss: %.3f" % hloss)

    return conf_matrix, accuracy, lr_auc, hloss

def classification_metrics_ml(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels, pred_labels)
    print("Accuracy: %.3f" % accuracy)

    print(classification_report(true_labels, pred_labels, target_names=['a', 'b','c','d','e']))
    print("Confusion Matrix:")
    conf_matrix = multilabel_confusion_matrix(true_labels, pred_labels)

    print(str(conf_matrix))

    lr_auc = roc_auc_score(true_labels, pred_labels, multi_class='ovr')
    print(' aucROC=%.3f' % (lr_auc))

    hloss = hamming_loss(true_labels, pred_labels)
    print("Hamming loss: %.3f" % hloss)

    return conf_matrix, accuracy, lr_auc, hloss

def main(args):
    # print version of the framework and GPUs
    print("Tensorflow Version: " + tf.__version__)
    print("GPUs available: " + str(len(tf.config.list_physical_devices('GPU'))))
    print("CPUs available: " + str(os.cpu_count()))
    channels = args.channels
    repeat = False
    if args.model in ['vgg', 'mobilenet', 'inception']:
        repeat = True

    batch_size = args.batch_size
    # load the model
    if args.bnn_build:

        # TODO make this a method shared between train and saved
        # set loss and final activation functions based on model type
        if args.model_type == "multi_class":
            loss_fcn = get_loss("crossentropy", args.label_smooth)
            activation_fcn = "softmax"
        elif args.model_type == "multi_label":
            loss_fcn = get_loss("binarycrossentropy", args.label_smooth)
            activation_fcn = "sigmoid"
        else:
            sys.exit("Specificed model type not allowed")

        # calculate input shape parameters
        # calculate input shape parameters
        step = (args.overlap/100) * \
            ((args.win_size * 0.001) * args.sample_rate)
        time_axis = (int)((args.duration * args.sample_rate) // step)
        # stft frequency axis is different than MFCC, time axises are the same
        if args.model_input == 'stft':
            freq_axis = (args.sample_pts//2) + 1
        else:
            freq_axis = args.mel_bins
        input_shape = (time_axis, freq_axis, channels)

        gpus = len(tf.config.list_physical_devices('GPU')) 
        if gpus > 1:
            print("There are multiple GPUs")
            print(gpus)
            # need to adjust batch size since each GPU gets batch size / num gpus data
            # this is necessary for calculating step size later as each gpu gets args.batch_size / num gpus samples per batch
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = build_model(args, input_shape, activation_fcn, 1)

                # compile model
                model.compile(optimizer=get_optimizer(args),
                            loss=loss_fcn,
                            metrics=[args.eval_metrics])

                model.load_weights(args.saved_model_path)

        else:
            model = build_model(args, input_shape, activation_fcn, 1)

            # compile model
            model.compile(optimizer=get_optimizer(args),
                        loss=loss_fcn,
                        metrics=[args.eval_metrics])

            model.load_weights(args.saved_model_path)
    else:
        model = tf.keras.models.load_model(args.saved_model_path)
    print(model.summary())


    # Load data for model predictions
    #random.seed(42) this is now completed in AudioDataProvider
    #mldata = AudioDataProvider(args)
    
    #mldata.describe()
    

    # If you just want to read in the dataset as a train test split

    start_date = datetime.datetime.strptime(args.start_date, '%Y%m%d %H%M%S')
    end_date = datetime.datetime.strptime(args.end_date, '%Y%m%d %H%M%S')

    if args.dataset=='tfrecord':
        #if args.test_data_type == 'original_split' or args.test_data_type == 'new_split':
        if args.test_data_type == 'original_split':
            mldata = AudioDataProvider(args)
            mldata.describe()
            test_files = mldata.get_data("test")
            #validation_files  = mldata.get_data("validation")
            #train_files = mldata.get_data("train")


        elif args.test_data_type == 'new_split':
            data_manager = DataHandler(args)
            train_files, validation_files, test_files = data_manager.make_dataset()

        elif args.test_data_type == 'data_dir':
            data_manager = DataHandler(args)
            test_files = data_manager.make_dataset()
            if len(test_files)==0:
                print("Stopping prediction, no available tfrecords")
                return
        
        test_data, test_metadata = get_audio_dataset(args, test_files, 1, args.data_type, 'features')
        print("\n###################\nTest Dataset Metadata\n###################\n")
        print(test_metadata)

        test_data = test_data.batch(args.batch_size,drop_remainder=True)
        

        plot_data, plot_metadata = get_audio_dataset(args, test_files, 1, args.data_type, 'features-times')
        plot_data = plot_data.batch(args.batch_size, drop_remainder=True)
        


        '''
            print("Creating Dataset")
            full_file_list = glob.glob(os.path.join(args.data_dir,"*.tfrecords"))
            file_list = []

            start_date = datetime.datetime.strptime(args.start_date, '%Y%m%d %H%M%S')
            end_date = datetime.datetime.strptime(args.end_date, '%Y%m%d %H%M%S')

            for fname in full_file_list:
                file_start_date, file_end_date = file_date(fname, 'second', 'both')

                if (start_date < file_start_date and file_start_date<=end_date<=file_end_date) or (file_start_date<=start_date<=file_end_date and file_end_date<end_date) or (file_start_date<=start_date<=file_end_date and file_start_date<=end_date<=file_end_date) or (start_date<=file_start_date and file_end_date<=end_date):
                    file_list.append(fname)
                else:
                    continue


            # Sort files by date
            file_list = sorted(file_list, key = file_date)

            if args.test_data_type == 'split':
        
                #test_files = mldata.get_data("test")
                #validation_files  = mldata.get_data("validation")
                #train_files = mldata.get_data("train")

                train_files = []
                validation_files = []
                test_files = []

                

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
            
            #train_data, train_metadata = get_audio_dataset(args, train_files, 1, args.data_type, 'features')
            test_data, test_metadata = get_audio_dataset(args, test_files, 1, args.data_type, 'features')
            #validate_data, val_metadata = get_audio_dataset(args, validation_files, 1, args.data_type, 'features')

            #print("\n###################\nTrain Dataset Metadata\n###################\n")
            #print(train_metadata)
            print("\n###################\nTest Dataset Metadata\n###################\n")
            print(test_metadata)
            #print("\n###################\nVal Dataset Metadata\n###################\n")
            #print(val_metadata)

        # If you want to use all data from a particular directory
        elif args.test_data_type == 'data_dir':
        
            # Use if you want to get all the files from a particular directory instead of the above to make prediction on full dataset
            #test_files = mldata.get_data("all")
            #test_data, test_metadata = get_audio_dataset(args, test_files, 1, args.data_type, 'features')

            test_data, test_metadata = get_audio_dataset(args, file_list, 1, args.data_type, 'features')

            #if args.print_specgram:
            plot_data, plot_metadata = get_audio_dataset(args, file_list, 1, args.data_type, 'features-times')
            plot_data = plot_data.batch(args.batch_size, drop_remainder=True)

            #test_data, test_metadata = get_audio_dataset(test_files, 1, 'mel', 'metadata')

            print("\n###################\nAll Metadata\n###################\n")
            print(test_metadata)


        test_data = test_data.batch(args.batch_size,drop_remainder=True)

        '''
        
        # run model.predict
        true_l = [] #np.array([])
        start_times = np.array([])
        end_times = np.array([])
        
        true_l = np.concatenate([y for x, y in test_data], axis=0)
        start_times = np.concatenate([start for x, y, start, end in plot_data], axis=0)
        end_times = np.concatenate([end for x, y, start, end in plot_data], axis=0)
        #id = np.concatenate([id for x, y, start, end, id in plot_data], axis=0)

    elif args.dataset == 'wav':
        data_proc = data_processor.AudioDataset(win_size=args.win_size, overlap=args.overlap, sample_points=args.sample_pts, mel_bins=args.mel_bins, dur=args.duration, modelType=args.model_type, 
                                            sample_rate=args.sample_rate, data_filepath=args.data_dir, epochs=1, batch_size=args.batch_size, mode=args.mode, repeat=repeat)
 
        if args.model_input == 'stft' and args.channels == 4:
            model_input_generator = 'stft_multi_channel'
        elif args.model_input == 'mfcc' and args.channels == 4:
            model_input_generator = 'mfcc_multi_channel'
        elif args.model_input == 'mfcc' and args.channels == 1:
            model_input_generator = 'mfcc'

        test_data, _, _ = data_proc.make_dataset('test_labels.csv', model_input_generator)

        true_l = data_proc.get_true_labels()

        labels = {'classA-':0, 'classB-':1, 'classC-':2, 'classD-':3, 'classE-':4}
        true_l = [labels[x[0]] for x in true_l]
        true_l = tf.keras.utils.to_categorical(true_l, num_classes=5)
        #print(true_l)

    # save spectrograms
    # this aids in verifying the data pipeline is correct and that labels are correctly matched to
    # spectrograms, also helps visualize dataset
    # see spectrogram_generator.py for more

    # calculate input shape parameters
    step = (args.overlap/100 )* ((args.win_size * 0.001) * args.sample_rate)
    time_axis = (int)((args.duration * args.sample_rate) // step)
    # stft frequency axis is different than MFCC, time axises are the same
    if args.model_input == 'stft':
        freq_axis = (args.sample_pts//2) +1
    else:
        freq_axis = args.mel_bins 

    if args.print_specgram: 
        print("Generating Spectrograms")
        for spec_batch, label_batch in test_data.take(1):
            generate_from_model(args, spec_batch, label_batch,[],[],time_axis, freq_axis)
        print("Spectrograms saved")

    '''
    # Get metadata for times and positions to create plots with
    for x, y, a, b, c, d, e, start_time, end_time, positions, f, g, h in test_data:
        true_l = np.append(true_l, y, axis=0)
        start_times = np.append(start_times, start_time, axis=0)
        end_times = np.append(end_times, end_time, axis=0)

        # Grab positions and store in dataframe, discard duplicates to plot
    '''

    # Make predictions on the data
    if args.bnn:
        print('Starting ' + str(args.num_mc_inference) + ' MC inferences')
        probs = tf.stack([model.predict(test_data, verbose=1)
                          for _ in range(args.num_mc_inference)], axis=0)
        preds = np.swapaxes(probs.numpy(),0,1)
        pred_vi, pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi, var, norm_entropy_vi, epistemic, aleatoric = BNN_predict(5,preds)
        predict_labels = pred_max_p_vi
        categorical_predict_labels = tf.keras.utils.to_categorical(predict_labels, num_classes=5)
    else:
        print("Making prediction")
        probs = model.predict(test_data)
        predict_labels = probs.argmax(axis=-1)
        categorical_predict_labels = tf.keras.utils.to_categorical(predict_labels, num_classes=5)

    # Save the predictions to a pickle file
    print(probs.shape)
    toSave = {
        'preds': probs,
        'trueLabels': true_l
    }
    print(toSave.keys())
    file = open(os.path.join(args.checkpoint_dir, "bnn.pkl"), 'wb')
    pickle.dump(toSave, file)
    file.close()
    #import ipdb;
    #ipdb.set_trace()

    # Save predictions to csv

    if args.dataset == 'tfrecord':
        if args.bnn:
            filepath_df = pd.DataFrame(zip(true_l, predict_labels, entropy_vi, pred_std_vi, pred_max_p_vi, pred_vi_mean_max_p, nll_vi, var, norm_entropy_vi, epistemic, aleatoric, start_times, end_times),columns=['true_l', 'predict_labels', 'entropy_vi', 'pred_std_vi', 'pred_max_p_vi', 'pred_vi_mean_max_p', 'nll_vi', 'var', 'norm_entropy_vi', 'epistemic', 'aleatoric', 'start_time', 'end_time'])
            filepath_df.to_csv(os.path.join(args.checkpoint_dir,"verified_samples.csv"), index=False)
        else:
            filepath_df = pd.DataFrame(zip(true_l, categorical_predict_labels, start_times, end_times), columns=['true_l','predict_labels','start_time','end_time'])
            filepath_df.to_csv(os.path.join(args.checkpoint_dir,"verified_samples.csv"), index=False)

    elif args.dataset == 'wav':    
        if args.bnn:
            filepath_df = pd.DataFrame(zip(true_l, predict_labels, entropy_vi, pred_std_vi, pred_max_p_vi, pred_vi_mean_max_p, nll_vi, var, norm_entropy_vi, epistemic, aleatoric),columns=['true_l', 'predict_labels', 'entropy_vi', 'pred_std_vi', 'pred_max_p_vi', 'pred_vi_mean_max_p', 'nll_vi', 'var', 'norm_entropy_vi', 'epistemic', 'aleatoric'])
            filepath_df.to_csv(os.path.join(args.checkpoint_dir,"verified_samples.csv"), index=False)
        else:
            filepath_df = pd.DataFrame(zip(true_l, categorical_predict_labels), columns=['true_l','predict_labels'])
            filepath_df.to_csv(os.path.join(args.checkpoint_dir,"verified_samples.csv"), index=False)

    if args.print_specgram: 
        # Create ais 
        pre_plot_data, pre_plot_metadata = get_audio_dataset(args, test_files, args.batch_size, args.data_type, 'metadata')
        
        batch_size = int(pre_plot_metadata['Examples Count'])

        new_ais_df = format_ais(args, pre_plot_data, batch_size)

        ais_graph(args, start_date, end_date, new_ais_df, preds=filepath_df)
        

    if args.model_type == 'multi_class':           
        print('Report Bayesian AUC')
        #predict_labels = tf.keras.utils.to_categorical(predict_labels, num_classes=5)
        #if args.dataset == 'tfrecord':
        true_l = true_l.argmax(axis=-1)
        #elif args.dataset == 'wav':  
        #    true_l = true_l

        #print(true_l)
        #print(categorical_predict_labels)
        conf_matrix, accuracy, lr_auc, hloss = classification_metrics_mc(true_l, categorical_predict_labels)

    if args.model_type == 'multi_label':
        pred_vi_working = copy.deepcopy(pred_vi)
        for item in pred_vi_working:
            if item[4] >= 0.5:
                #if predictive probablitly of class E is >= to .5, get the index (which will correspond to class)
                # of the highest predictive probability
                k = np.argmax(item)
                if k == 4:
                    #if the highest probability is class E, predict class E and nothing else
                    for i in range(len(item)):
                        item[i] = 0
                    item[4] = 1
                else:
                    #if class E is not the highest, don't predict it, predict everything else that is over 0.5
                    item[4] = 0
                    for i in range(len(item)):
                        if item[i] >=0.5:
                            item[i] = 1
                        else:
                            item[i] = 0
            elif  max(item) < 0.5:
                #if no class has a predictive probability over 0.5, just predict the highest
                k = np.argmax(item)
                for i in range(len(item)):
                    item[i] = 0
                item[k] = 1
            else:
                for i in range(len(item)):
                    if item[i] >=0.5:
                        item[i] = 1
                    else:
                        item[i] = 0
        predict_labels = pred_vi_working
        conf_matrix, accuracy, lr_auc, hloss = classification_metrics_ml(true_l, predict_labels)


    # Old way of doing it
    '''
    if args.model_type == 'multi_class':
        preds = np.swapaxes(probs.numpy(),0,1)
        #preds = np.swapaxes(probs)
        pred_vi, pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi, var, norm_entropy_vi, epistemic, aleatoric = BNN_predict(5,preds)
        pred_labels = pred_vi.argmax(-1)
        #filepaths = data_proc.get_filepaths()
        #filepath_df = pd.DataFrame(zip(filepaths, pred_max_p_vi, entropy_vi, pred_std_vi, pred_max_p_vi, pred_vi_mean_max_p, nll_vi, var, norm_entropy_vi, epistemic, aleatoric))
        predict_labels = pred_max_p_vi
        filepath_df = pd.DataFrame(zip(true_l, predict_labels, entropy_vi, pred_std_vi, pred_max_p_vi, pred_vi_mean_max_p, nll_vi, var, norm_entropy_vi, epistemic, aleatoric))
        filepath_df.to_csv(os.path.join(args.checkpoint_dir,"verified_samples.csv"), index=False)
    if args.model_type == 'multi_label':
        preds = np.swapaxes(probs.numpy(),0,1)

        pred_vi, pred_max_p_vi, pred_vi_mean_max_p, entropy_vi, nll_vi, pred_std_vi, var, norm_entropy_vi, epistemic, aleatoric = BNN_predict(5,preds)
        pred_vi_working = copy.deepcopy(pred_vi)
        for item in pred_vi_working:
            if  max(item) < 0.5:
                k = np.argmax(item)
                for i in range(len(item)):
                    item[i] = 0
                item[k] = 1
            else:
                for i in range(len(item)):
                    if item[i] >=0.5:
                        item[i] = 1
                    else:
                        item[i] = 0
        #filepaths = data_proc.get_filepaths()
        #filepath_df = pd.DataFrame(zip(filepaths, pred_vi_working, entropy_vi, pred_std_vi, pred_max_p_vi, pred_vi_mean_max_p, nll_vi, var, norm_entropy_vi, epistemic, aleatoric))
        predict_labels = pred_vi_working
        filepath_df = pd.DataFrame(zip(true_l, predict_labels, entropy_vi, pred_std_vi, pred_max_p_vi, pred_vi_mean_max_p, nll_vi, var, norm_entropy_vi, epistemic, aleatoric))
        filepath_df.to_csv(os.path.join(args.checkpoint_dir,"verified_samples.csv"), index=False)
    '''

    '''
    #else:
    predict_probs = model.predict(test_data)
    predict_labels = predict_probs.argmax(axis=-1)

    if args.save_predictions:
        filepath_df = pd.DataFrame(zip(true_l, predict_labels))
        filepath_df.to_csv('verified_samples.csv', index=False)
    

    class_labels = args.classes.split(',')
    '''

    # Calculate performance metrics

    '''

    if args.model_type == 'multi_class':
        # True labels are in word form, need to convert to numbers
        label_encode = LabelEncoder()
        true_labels = label_encode.fit_transform(true_l)

        # plot useful outputs
        # get confusion matrix, 2D numpy array, using tensorflow built in function
        print("Confusion Matrix:")
        conf_matrix = confusion_matrix(true_labels, predict_labels)
        print(str(conf_matrix))
        print(classification_report(true_labels,
                                    predict_labels, target_names=class_labels))
        # get accuracy
        accuracy = accuracy_score(true_labels, predict_labels)
        print("Accuracy: %.3f" % accuracy)

    elif 'regression' in args.model_type:
        # Difference evaluation metrics for regression, no need to encode labels
        mse = mean_squared_error(true_l, predict_labels)
        mae = mean_absolute_error(true_l, predict_labels)
        print("REGRESSION:")
        print("Mean Squared Error: %.3f" % mse)
        print("Mean Absolute Error: %.3f" % mae)

        # this is just a starter, more or different meterics will be required in the future.

    elif args.model_type == 'multi_label':
        # multi-label requires different metrics
        prediction = tf.cast(predict_probs, tf.float32)
        threshold = float(0.5)
        predict_labels = tf.cast(tf.greater(prediction, threshold), tf.int64)
        # True labels are in word form, need to convert to numbers
        label_encode = MultiLabelBinarizer()
        true_labels = label_encode.fit_transform(true_l)

        ham_loss = hamming_loss(true_labels, predict_labels)
        print("Hamming Loss: %.3f" % ham_loss)

        avg_percision = average_precision_score(true_labels, predict_labels)
        print("mean Average Precision: %.3f" % avg_percision)

        auc_score = roc_auc_score(true_labels, predict_labels)
        print("AUC: %.3f" % auc_score)

        print(classification_report(true_labels,
                                    predict_labels, target_names=class_labels))

        '''






    # Old code way of calculating metrics
    '''
    #else:
    predict_probs = model.predict(test_data)
    predict_labels = predict_probs.argmax(axis=-1)

    if args.save_predictions:
        filepath_df = pd.DataFrame(zip(filepaths, predict_labels))
        #filepath_df = pd.DataFrame(saved_fps)
        filepath_df.to_csv('verified_samples.csv', index=False)

    class_labels = args.classes.split(',')

    if args.model_type == 'multi_class':
        # True labels are in word form, need to convert to numbers
        label_encode = LabelEncoder()
        true_labels = label_encode.fit_transform(true_l)

        # plot useful outputs
        # get confusion matrix, 2D numpy array, using tensorflow built in function
        print("Confusion Matrix:")
        conf_matrix = confusion_matrix(true_labels, predict_labels)
        print(str(conf_matrix))
        print(classification_report(true_labels,
                                    predict_labels, target_names=class_labels))
        # get accuracy
        accuracy = accuracy_score(true_labels, predict_labels)
        print("Accuracy: %.3f" % accuracy)

    elif 'regression' in args.model_type:
        # Difference evaluation metrics for regression, no need to encode labels
        mse = mean_squared_error(true_l, predict_labels)
        mae = mean_absolute_error(true_l, predict_labels)
        print("REGRESSION:")
        print("Mean Squared Error: %.3f" % mse)
        print("Mean Absolute Error: %.3f" % mae)

        # this is just a starter, more or different meterics will be required in the future.

    elif args.model_type == 'multi_label':
        # multi-label requires different metrics
        prediction = tf.cast(predict_probs, tf.float32)
        threshold = float(0.5)
        predict_labels = tf.cast(tf.greater(prediction, threshold), tf.int64)
        # True labels are in word form, need to convert to numbers
        label_encode = MultiLabelBinarizer()
        true_labels = label_encode.fit_transform(true_l)

        ham_loss = hamming_loss(true_labels, predict_labels)
        print("Hamming Loss: %.3f" % ham_loss)

        avg_percision = average_precision_score(true_labels, predict_labels)
        print("mean Average Precision: %.3f" % avg_percision)

        auc_score = roc_auc_score(true_labels, predict_labels)
        print("AUC: %.3f" % auc_score)

        print(classification_report(true_labels,
                                    predict_labels, target_names=class_labels))
    '''

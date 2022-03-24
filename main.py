'''
 Andrew Pfau
 Sonar Classifier
 This is the started script for all model operations. It parses command line arguments and
 then saves them to a yaml file in the checkpoints directory. 
 The mode parameter determines which script will be called next, either to train a model
 from scratch or to load saved model parameters.
 
'''
# import libraries
import argparse
import os
import sys
import json
import datetime

# imports from other files in project
# import train_model
import saved_model
import train_model_k_fold
import tensorflow as tf
import train_model as train_model
import plots

# for checking user input
from models.model import model_dict
#tf.compat.v1.disable_eager_execution()

def main():
    args = parse_args()
    
    sanity_checker(args)
    # make dir
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    # save hyperparameters to yaml file
    save_params(args)
    
    # switch on mode
    if args.mode == 'train' or args.mode == 'cont-train' or args.mode == 'active-train':
        train_model.main(args)
    elif args.mode == 'saved':
        saved_model.main(args)
    elif args.mode == 'train-k-fold':
        train_model_k_fold.main(args)
    elif args.mode == 'plot':
        plots.main(args)
    else:
        sys.exit("WARNING: mode given is not a valid option!")
    '''
    elif args.mode == 'active':
        saved_model_active = Saved_Model_Active(args)
        saved_model_active.main(args)
    '''

def sanity_checker(args):
    """
    This function will check argument pairings to make sure that everything is correct since there a many arguments
    and it could be easy to miss one and then have things misconfigured
    """
    if 'regression' in args.model_type:
        if args.channels == 1:
            sys.exit("WARNING: attempting regression with only 1 input channel. There must be more than one channel for regression. Please change and try again.")
        if args.model_type == 'regression_bearing_range' and args.num_classes != 2:
            sys.exit("WARNING: attempting multi output regression without 2 outputs. Please change num_classes to 2 and try again.")
        if args.model_type in ['regression_bearing', 'regression_range' ] and args.num_classes != 1:
            sys.exit("WARNING: attempting regression with more than 1 output. Please change num_classes to 1 and try again.")

    if args.mode == 'saved' and args.saved_model_path == None:
        sys.exit("WARNING: attempting to run a saved model without providing an input path. Please try again with a saved_model_path.")

    if args.checkpoint_dir == None:
        sys.exit("WARNING: no checkpoint directory given. Please specify a checkpoint_dir and try again.")

    if args.model_type not in ['multi_class', 'multi_label', 'regression_bearing', 'regression_range', 'regression_bearing_range']:
        sys.exit("WARNING: invalid model type. Valid oprtions are: multi_class, multi_label, regression_bearing, regression_range, and regression_bearing_range. Please try again.")

    if args.model not in model_dict.keys():
        sys.exit("WARNING: invalid model. Valid oprtions are: "+ str(model_dict.keys()) +" Please try again.")


def parse_args():
    """
    This function parses command line arguments to the main.py program
    --help , -h : help lists all arguments and a description

    return: returns a parse args object
    """
    str_to_bool = lambda x : True if x.lower() == 'true' else False
    parser = argparse.ArgumentParser()
    # data processing, data_dir and checkpoint_dir are required arguments, saved_model_path if in 'saved' mode
    parser.add_argument('--read_from_file',      type=str_to_bool,   default='false',    help="If true, read in arguments from file specified in arg_file argument")
    parser.add_argument('--arg_file',            type = str,         default='params.txt', help="Path to arguments file")
    
    parser.add_argument('--data_dir',         type=str, help="Directory of data files. This directory should contatin the train, test, and validate fodlers")
    parser.add_argument('--checkpoint_dir',   type=str, help="Directory to store checkpoint files and output.")
    parser.add_argument('--saved_model_path', type=str, help='Path to model checkpoint file' )
    parser.add_argument('--mode',             type=str, default='train', help="Mode to operate in, either train,saved, or train-k-fold. Train will train a new model, cont-train will continue training from a previous checkpoint,\
                                                                        Saved will load the model specificed in saved_model_path. Train-k-fold will train the model with k fold cross validation, must specify the k parameter")
    parser.add_argument('--classes',          type=str, default='classA, classB, classC, classD, classE', help="Comma seperated list of classes, only used during saved model prediction.") 

    # spectrogram parameters
    parser.add_argument('--model_input',      type=str, default='mfcc', help="Input format into model, either stft or mfcc.")
    parser.add_argument('--win_size',         type=int, default=250,    help='Spectrogram window size in msec')
    parser.add_argument('--overlap',          type=int, default=75,     help='Spectrogram window overlap in percent')
    parser.add_argument('--sample_pts',       type=int, default=1024,    help='Number of FFT sample points')
    parser.add_argument('--sample_rate',      type=int, default=4000,   help='Sample rate of input audio')
    parser.add_argument('--duration',         type=int, default=30,     help='Audio duration in seconds')
    parser.add_argument('--channels',         type=int, default=4,      help='Number of input channels, Default is 1')
    
    # mfcc specific parameter, still requires the above spectrogram params as well
    parser.add_argument('--mel_bins',         type=int, default=128,     help="Number of Mel Frequency bins, only used in model_input parameter is mfcc")

    # model parameters
    parser.add_argument('--num_epochs',          type=int,   default=2,               help='Number of training epochs')
    parser.add_argument('--batch_size',          type=int,   default=256,             help='Number of examples per batch')
    parser.add_argument('--optimizer',           type=str,   default='sgd',          help="Optimizer to use during training")
    parser.add_argument('--learning_rate_start', type=float, default=0.001,           help="Starting learning rate")
    parser.add_argument('--eval_metrics',        type=str,   default="accuracy",      help="Model evaluation metric")
    parser.add_argument('--callbacks',           type=str,   default='early_stop',    help="Comma seperated list of callbacks to use")
    parser.add_argument('--model',               type=str,   default="simple",        help="Model to use, from the model.py file")
    parser.add_argument('--model_type',          type=str,   default="multi_class",   help="Model type to use, either multi_class, multi_label, or regression, determines loss function and activation function")
    parser.add_argument('--num_classes',         type=int,   default=2,               help="Number of classes/labels for classification output.")
    parser.add_argument('--label_smooth',        type=float, default=0.0,             help="Input to Categorical Crossentropy loss function label smoothing")
    parser.add_argument('--k',                   type=int,   default=5,               help="Number of folds for k fold cross validation.")
    parser.add_argument('--resnet_depth',        type=int,   default=44,              help="Depth of Resnet model, either 20, 44, or 56")
    
    parser.add_argument('--class_weight',        type=str_to_bool,   default='false',    help="If true class weight will be calculated from dataset and used.")
    parser.add_argument('--print_specgram',      type=str_to_bool,   default='false',    help="If true will generate spectrogram images from input pipeline.")
    parser.add_argument('--data_aug',            type=str_to_bool,   default='false',    help="If true will include data augmentation in data pipeline, Defaul is False")
    parser.add_argument('--save_predictions',    type=str_to_bool,   default='false',    help="If true save prediction output of saved_model to verified_samples.csv. Default is Fase")
    

    parser.add_argument('--num_mc_inference',    type=int,   default=50,              help="Num of MC inferences to run")
    parser.add_argument('--kl_term',             type=int,   default=-1,              help="KL divergence scale, -1 = auto")
    parser.add_argument('--bnn_type',            type=str,   default='none',          help="BNN types used during training")
    parser.add_argument('--bnn',                 type=str_to_bool,   default='false', help="True to run BNN predictions, default is false")
    parser.add_argument('--mc_dropout_prob',     type=float, default=0.35,             help="Dropout probability in a MC dropout model")
    parser.add_argument('--bnn_build',           type=str_to_bool,   default='false', help="set to true when running saved TFP model")
    parser.add_argument('--queryMethod',            type=str,   default='MaxEntropy',          help="BNN types used during training")           
    parser.add_argument('--trainingFile',            type=str,   default='train_labels.csv',          help="BNN types used during training")   
    parser.add_argument('--dropout_l2', type=float, default=0.0001,           help="Starting learning rate")

    parser.add_argument("--resnet_version",     type=int,default=1, help="specify resnet model version")
    parser.add_argument("--resnet_n",           type=int,default=3, help="resnet model parameter, see bellow")
    parser.add_argument("--kl_weight",           type=float,default=1.0, help="resnet model parameter, see bellow")
    parser.add_argument("--active_loops",           type=int,default=100, help="resnet model parameter, see bellow")
    parser.add_argument("--test_data_type",           type=str,default='split', help="split to use test split from training or data_dir if pointing at a directory to use for testing")
    parser.add_argument("--start_date",           type=str,default='19700101', help="Define start date to use data for if want to use subset of data in data_dir in format YYYYMMDD")
    parser.add_argument("--end_date",           type=str,default='30000101', help="Define end date to use data for if want to use subset of data in data_dir in format YYYYMMDD")
    parser.add_argument("--data_shuffle",           type=str_to_bool, default='true', help="Whether you want the data shuffled or not")
    parser.add_argument("--active_time_step",           type=str, default='1', help="How many days before retraining model on active loop")
    parser.add_argument("--plot_type",           type=str, default='accuracy', help="Type of plot to make. accuracy, spectrogram,")
    parser.add_argument("--database",           type=str, default='/group/mbari/upload/MARSInfo/master_index.db', help="path to database for acoustic data")
    parser.add_argument("--segment_length",           type=int, default=30, help="Length of each segment in query for acoustic data")
    parser.add_argument("--data_type",           type=str, default='mel', help="Format of the data being used, can be raw, calibrated, mel, and calibrated-mel")
    parser.add_argument("--seed",           type=int, default=42, help="seed to use when shuffling datasets")
    parser.add_argument("--shuffle_group",           type=str, default='day', help="group to initially shuffle files by")
    parser.add_argument("--dataset",           type=str, default='tfrecord', help="specify is using the tfrecord or wav older dataset")
    args =  parser.parse_args()

    # if reading in args from a file, overwrite the arguments now with those from the file
    if args.read_from_file:
        t_args = argparse.Namespace()
        with open(args.arg_file, 'r') as f:
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    
    return args

def save_params(params):
    '''
    Save model parameters from the command line to a file called params.txt.
    Params are written just as they are from the command line so that argparse can read them back in from a file.
    '''
    path_ = os.path.join(params.checkpoint_dir, 'params.txt')
    with open(path_, 'w') as f:
        json.dump(params.__dict__, f, indent=2)   


if __name__ == "__main__":
    main()

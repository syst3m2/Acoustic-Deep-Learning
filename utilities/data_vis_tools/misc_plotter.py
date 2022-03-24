"""
Andrew Pfau
Miscellaneous plotting tool
"""

import numpy as np
import os
import sys
import glob
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import pickle
#import seaborn as sn


def plotting_fcn(mode, data_file, save_filename):
    # input must be a csv file
    data = pd.read_csv(data_file, header=0, delimiter=',')
    
    # set larger figure size
    plt.figure(figsize=(10,10))

    if mode == 'cpa':
        cpa_data = data['CPA'].to_numpy()
        # use the built in matplotlib hist function to make a histogram plot
        plt.hist(cpa_data, bins=21, ec='black')
        plt.title("CPA RANGE HISTOGRAM (KM)")
        plt.xticks(np.arange(0,21,2))
        plt.xlabel('CPA RANGE (KM)')
        plt.ylabel('Frequency')

    elif mode == 'bearing':
        # plot historgram of bearings in linear plot from 0(north) to 359
        # use the full_list_labels.csv file where the 'LABEL' column is a comma seperated list of labels, with bearing being element 1
        bearing_data_df = data['LABEL']
        bearing_data = []
        for row in bearing_data_df:
            labels = row.split(',')
            if len(labels) > 1:
                brg = labels[1]
                brg = brg.replace(']','')
                bearing_data.append(float(brg))
        
        # want to plot a histogram where each bin is 10 degrees of bearing
        plt.hist(bearing_data, bins=36, ec='black')

        plt.title("BEARING HISTOGRAM")
        plt.xticks(np.arange(0,360, 20))
        plt.xlabel('TRUE BEARING')
        plt.ylabel('Frequency')

    else:
        sys.exit("WARNING: mode: "+str(mode)+" not supported!")

    plt.savefig(save_filename)
    plt.clf()

def training_curves_plotter(tgt_log, title, save_name, plot_type):
    """
    Plot accuracy or loss curves from training. Plots both training and validation data.
    Args:
        tgt_log:   Data log, usually log.csv from model training 
        title:     Title for plot
        save_name: File name for plot ouput file, can include path
        plot_type: Type of plot, either 'loss' or 'accuracy'

    """
    cwd = os.getcwd()
    data = pd.read_csv(os.path.join(cwd, tgt_log), header=0, delimiter=';')
    y_label = ""
    
    # x data always the same
    x_data = data['epoch']
   
    plt.title(title)
    plt.figure(figsize=(10,10))

    if plot_type == "accuracy" or plot_type == 'all':    
        plt.plot(x_data, data['accuracy'], c='green', label='Train Accuracy')
        plt.plot(x_data, data['val_accuracy'], c='blue', label='Validate Accuracy')

    if plot_type == "loss" or plot_type == "all":
        plt.plot(x_data, data['loss'], c='orange', label='Train Loss')
        plt.plot(x_data, data['val_loss'], c='purple', label='Validate Loss')

    if plot_type == 'learning rate' or plot_type == 'all':
        plt.plot(x_data, data['lr'], c='red', label="Learning Rate" ) 


    plt.legend(loc="lower right")
    plt.grid(True)
    plt.xlabel("Epoch")

    plt.savefig(save_name)
    plt.close()

def main():
    """
    Main function called from the command line, parse command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',        type=str, help="Program mode. Options are 'histogram' or 'training'.")
    parser.add_argument('--plot_type',   type=str, help="Plot data type. For Histograms, options are cpa or bearing. For training curves, options are accuracy, loss, learing rate, or all")

    parser.add_argument('--data_file',   type=str, help="Path to csv file containing data to plot. Expected format is ?")
    parser.add_argument('--output_file', type=str, help="Output plot filename",                     default='output.png')

    args = parser.parse_args()

    if args.mode.lower() == 'histogram':
        if args.plot_type not in ['cpa', 'bearing']:
            sys.exit("WARNING: Plot type given not an option")
        plotting_fcn(args.plot_type, args.data_file, args.output_file)
    elif args.mode.lower() == 'training':
        if args.plot_type not in ['all', 'learning rate', 'loss', 'accuracy']:
            sys.exit("WARNING: Plot type given not an option")
        training_curves_plotter(args.data_file, "title", args.output_file, args.plot_type)
    else:
        sys.exit("WARNING: Mode given not an option")

if __name__ == "__main__":
    main()
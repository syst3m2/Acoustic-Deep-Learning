"""
Andrew Pfau
Sonar Classifier

    This program is used to generate plots of data including plots used in power point briefs and papers
    It plots ship range vs time and places colored dots for each sample classified

    It requires 2 input files, one with ground truth time and range from AIS data
    a second file is the output of a classifier. In order to get this data, run saved_model with the --save_predictions
    flag set to true.

    There is also a function to plot training and loss curves from log.csv files from model training


"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, HourLocator
import os
import datetime
import sys
import argparse

# Used for multi ship plotting
def range_plotter_2(data_log, predicts_log):
    allData = pd.read_csv(data_log,  header=0, parse_dates=[0], names=['TIME', 'MMSI', 'RNG'])
    labelData = pd.read_csv('range_plots_AIS/03-08-multi-label-results-devModel.csv',  header=0, names=['FNAME', 'LABEL'])
    #print(labelData.head())
    
    label_list = []
    for line in labelData.iterrows():
        temp = line[1][1].replace('[','')
        temp = temp.replace(']','')
        temp = [int(s) for s in temp.split(' ')]
        label_list.append([idx for idx, ele in enumerate(temp) if ele])

    
    #exit()
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(15,15))
    # format x axis
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%m'))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))
    plt.gca().xaxis.set_tick_params(rotation = 30)
    
    plt.title("Multi Labeling in Multi Taget Env")
    # plot range vs time

    point_A_x = []
    point_A_y = []
    point_B_x = []
    point_B_y = []
    point_C_x = []
    point_C_y = []
    point_D_x = []
    point_D_y = []

    ships_time = {237916000:[], 240275000:[], 259770000:[], 303849000:[], 309484000:[], 311062400:[], 311433000:[]}
    ships_rng = {237916000:[], 240275000:[], 259770000:[], 303849000:[], 309484000:[], 311062400:[], 311433000:[]}
    for line in allData.iterrows():
        ships_time[line[1]['MMSI']].append(line[1]['TIME'])
        ships_rng[line[1]['MMSI']].append(line[1]['RNG'])
        
    for ship in ships_time:
        plt.plot(ships_time[ship], ships_rng[ship],c='black')
    
    plt.xlim([datetime.datetime(2013, 2, 8, 0, 0, 0), datetime.datetime(2013, 2, 8, 7, 0, 0)])
    plt.ylim(0,30.0)
    plt.savefig('output.png')
    plt.close()
    exit()

    plt.plot(contain1['TIME'], contain1['RNG'],c='blue')
    plt.plot(contain2['TIME'], contain2['RNG'],c='blue')
    plt.plot(contain3['TIME'], contain3['RNG'],c='blue')
    plt.plot(cruise['TIME'], cruise['RNG'],c='red')
    # format x axis
    #ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))   #to get a tick every 15 minutes
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
    results_label = labelData['LABEL']
    results_x = []
    results_y = []
    color_dict = {0:'black', 1:'yellow', 2:'red', 3:'blue', 4:'green'}
    for idx, line in enumerate(labelData.iterrows()):
        hr = (int)(line[1]['FNAME'].split('_')[3])
        temp = line[1]['FNAME'].split('_')[4]
        time = (int)(temp.split('.')[0])
        min = time // 60
        sec = time % 60
        clip_time = pd.Timestamp(2013, 3, 8, hr, min, sec)
        #if (len(label_list[idx]) > 1):
        for c in label_list[idx]:
            if c == 0: 
                point_A_x.append(clip_time)
                results_idx = tug['TIME'].sub(clip_time).abs().idxmin()
                point_A_y.append(tug['RNG'][results_idx])
            if c == 1: 
                point_B_x.append(clip_time)
                results_idx = allData['TIME'].sub(clip_time).abs().idxmin()
                point_B_y.append(allData['RNG'][results_idx])
            if c == 2: 
                point_C_x.append(clip_time)
                results_idx = cruise['TIME'].sub(clip_time).abs().idxmin()
                point_C_y.append(cruise['RNG'][results_idx])
            if c == 3: 
                point_D_x.append(clip_time)
                results_idx = allContain['TIME'].sub(clip_time).abs().idxmin()
                point_D_y.append(allContain['RNG'][results_idx])

    #colors = [color_dict[x] for x in results_label]

    #results_y = [1 for x in range(len(results_x))]
    plt.scatter(point_A_x, point_A_y, alpha=1, c='blue', label='Predicted A')
    plt.scatter(point_B_x, point_B_y, alpha=1, c='yellow', label='Predicted B')
    plt.scatter(point_C_x, point_C_y, alpha=1, c='red', label='Predicted C')
    plt.scatter(point_D_x, point_D_y, alpha=1, c='green', label='Predicted D')

    plt.xlim([datetime.datetime(2013, 3, 8, 5, 0, 0), datetime.datetime(2013, 3, 8, 9, 30, 0)])
    black = mpatches.Patch(color='blue', label='Class A, one Tug')
    yellow = mpatches.Patch(color='yellow', label='Class B, no in samples')
    blue = mpatches.Patch(color='green', label='ClassD, 3 Container Ships')
    red = mpatches.Patch(color='red', label='Class C, 1 Cruise Ship')
    #green = mpatches.Patch(color='green', label='Class E, no ship present')
    
    #plt.legend(handles=[black, yellow, red, blue],loc = 'lower right')
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Range (km)")

    plt.savefig('output.png')
    plt.close()


def range_plotter(tgt_file, results_file, plt_title, save_name, max_rng, num_classes, true_label_class, start_hr, start_min, end_hr, end_min ):
    """
    Program to plot range vs time and classifications
    Assumes that class labels are integers
    Args:
        tgt_file: Ground truth data from AIS
        results_file: Prediction results of model classification
        plt_title: Title for plot
        save_name: File name for plot ouput file, can include path
        max_rng: Max value for y axis
        num_classes: Number of different classes predicted total
        start_hr: Integer start hr, for x axis bounds
        start_min: Integer start min, for x axis bounds
        end_hr: Integer end hr, for x axis bounds
        end_min: Integer end min, for x axis bounds

    """
    
    data = pd.read_csv(tgt_file, parse_dates=[0], header=0)
    x_data = data['TIME']
    y_data = data['RNG']
    results_data = pd.read_csv(results_file,  header=0, names=['FPATH', 'CLASS'])

    results_x = []
    results_y = []
    line_1 = results_data['FPATH'][2].split('_')
    yr = (int)('20' + line_1[2][:2])
    day = (int)(line_1[2][4:])
    mon = (int)(line_1[2][2:4])
    
    for line in results_data.iterrows():
        hr = (int)(line[1]['FPATH'].split('_')[3])
        temp = line[1]['FPATH'].split('_')[4]
        time = (int)(temp.split('.')[0])
        min = time // 60
        sec = time % 60
        clip_time = pd.Timestamp(yr, mon, day, hr, min, sec)
        results_x.append(clip_time)
        # put the prediction on the closest AIS time available.
        results_idx = data['TIME'].sub(clip_time).abs().idxmin()
        results_y.append(data['RNG'][results_idx])

    results_class = results_data['CLASS']
    colors = []
    true_pred = 0
    false_pred = [0 for _ in range(num_classes)]
    for class_num in results_class:
        if class_num == true_label_class:
            true_pred += 1
            colors.append(true_label_class)
        else:
            colors.append(class_num)
            false_pred[class_num] += 1

    #better time formatting
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(12,12))
    # format x axis
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%m'))
    plt.gca().xaxis.set_major_locator(HourLocator(interval=1))
    plt.gca().xaxis.set_tick_params(rotation = 30)
    
    plt.title(plt_title)
    # plot range vs time
    plt.plot(x_data, y_data,c='black')
    # plot classification events
    plt.scatter(results_x, results_y, alpha=1, s=50, c=colors,label='Predicted')
    plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes") 
    #########################################################
    # old code used to make legend
    #black = mpatches.Patch(color='black', label='True Range')
    #blue = mpatches.Patch(color='blue', label='classD: ' + str(true_pred))
    #red = mpatches.Patch(color='red', label='classC: ' + str(false_pred_0))
    #green = mpatches.Patch(color='green', label='classE: ' + str(false_pred_3))
    #yellow = mpatches.Patch(color='darkorange', label='classA: ' + str(false_pred_1))

    #plt.legend(handles=[black, blue, green, red, yellow],loc = 'lower right')
    #######################################################

    plt.ylim(0, max_rng)
    plt.xlim([datetime.datetime(yr, mon, day, start_hr, start_min, 0), datetime.datetime(yr, mon, day, end_hr, end_min, 0)])
    plt.xlabel("Time")
    plt.ylabel("Range (km)")

    plt.savefig(save_name)
    plt.close()

def main():
    """
    Main function called from the command line, parse command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',        type=str, help="Program mode. Options are 'range' for range plots and 'curve' for training curves")
    parser.add_argument('--target_log',  type=str, help="Data log for curve plots or AIS range data for range plots")
    parser.add_argument('--predicts',    type=str, help="CSV file of predictions for range plotting")
    parser.add_argument('--plot_title',  type=str, help="Title for output plot",                    default='DEFAULT TITLE')
    parser.add_argument('--output_file', type=str, help="Output plot filename",                     default='output.png')
    parser.add_argument('--plot_type',   type=str, help="For training curves, type of data, either 'accuracy' or 'loss'. Default is 'accuracy'", default='accuracy')
    parser.add_argument('--max_rng',     type=int, help="Max range (y value) for range plotting. Default is 30", default=30)
    
    parser.add_argument('--num_classes',     type=int, help="Max range (y value) for range plotting. Default is 30", default=30)
    parser.add_argument('--true_class',      type=int, help="Integer number of ", default=30)
    
    parser.add_argument('--start_hr',        type=int, help="Start hour for x axis. Default is 0", default=0)
    parser.add_argument('--start_min',       type=int, help="Start min for x axis. Default is 0", default=0)
    parser.add_argument('--end_hr',          type=int, help="End hour for x axis. Default is 0", default=0)
    parser.add_argument('--end_min',         type=int, help="End min for x axis. Default is 0", default=0)


    args = parser.parse_args()

    if args.mode == 'range':
        range_plotter(args.target_log, args.predicts, args.plot_title, args.output_file, args.max_rng, args.num_classes, args.true_class, args.start_hr, args.start_min, args.end_hr, args.end_min)
    elif args.mode == 'range2':
        range_plotter_2(args.target_log, args.predicts)
    else:
        sys.exit("WARNING, function not supported")


if __name__ == "__main__":
    main()
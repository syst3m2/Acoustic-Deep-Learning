"""
    Andrew Pfau
    Sonar Classifier

    This program reads in AIS stream data from a matlab .mat file and converts it to a csv file
    Not all data is taken from the .mat files, only MMSI, time and range. Not all AIS information is realiable
    as it depends on the operator to update thier own information

    There are 4 modes, single and many, mmsi, and multilabel.

    In the many mode, a csv file is generated for each corresponding .mat file and a csv if all found mmsis is made.
    This second file is used by mmsiScraper.py to find ship info from the web

    In the single mode, a single csv file is generated for all .mat files. 

    In mmsi mode, a specific mmsi number must be provided via the --mmsi argument. Only data related to this mmsi will be
    saved from .mat files in the target_dir

    In the multilabel mode, overlapping ship data is captured.
"""
import datetime as dt
import tables
import pandas as pd
import os
import sys
import glob
import argparse

import warnings

def convertTime(time):
    """
        Helper function to convert matlab time format into python time obj
        Args:
            time: matlab time object
        Returns:
            Python time object representation without microseconds
    """
    day = dt.datetime.fromordinal(int(time))
    dayfrac = dt.timedelta(days=time%1) - dt.timedelta(days=366)
    t = day + dayfrac
    # return final time, set microseconds to 0
    return t.replace(microsecond=0)

def generate_single_output(data_dir, mmsiDB_file, output_filename, brg_and_rng, cpa_rng_beyond, cpa_rng_max, cpa_rng_min=0.0):
    """
    This function reads through all .mat AIS files in the data_dir and provides a single output
    It also uses data from the mmsiDB file to get ship type infomation
    If collecting bearing and range data, a pickle file is saved with all of the data as it is easier to reload a pickle file later
    The name of the pickle file will be the same as that given in the output_filename argument

    #Arguments:
        data_dir:        path to AIS .mat files to process
        mmsiDB_file:     path to csv file contatining mmsi and ship info
        output_filename: path and name for output csv
        brg_and_rng:     flag to indicate whether or not to collect bearing and range data.
        cpa_rng_beyond:  cpa range beyond which no ship is considered present in the data
        cpa_rang_min:    minimum cpa range in kilometers from sensor, usually 0  
        cpa_rang_max:    maximum cpa range in kilometers from sensor, usually set by user as the upper bound of
        ships to return
    """
    cwd = os.getcwd() 
    mmsiDB = pd.read_csv(os.path.join(cwd, mmsiDB_file), header=0)
    all_ships = pd.DataFrame()
    # format is key = mmsi, values = start_time, end_time, label, desig
    daily_ships ={} 
    find_no_ships = {}
    
    column_name = ["MMSI","START_TIME", "END_TIME", "LABEL", "DESIG", "CPA", "CPA_TIME" ]
    if brg_and_rng:
        column_name = ["MMSI","START_TIME", "END_TIME", "LABEL", "DESIG", "CPA", "CPA_TIME", "BEARING_RANGE_DATA"]
    
    for f in glob.glob(data_dir+'*.mat'):
        # store in key = time stamp, value = list of MMSI, range, brg
        with warnings.catch_warnings():  # this suppresses an HDFS datatype warning that is from a particular column in all .mat files
            warnings.simplefilter("ignore")
            h5file = tables.open_file(f, mode = 'r')
            for i, ship in enumerate(h5file.root):
                if i > 0:
                    mmsi = int(ship.MMSI[0][0])
                    for x, time in enumerate(ship.datenumber):
                        record_time = convertTime(time[0])
                        rng = round(ship.range[x][0], 3)
                        spd = int(ship.SOG[x][0])
                        brg = round(ship.bearing[x][0], 3)
                        # add mmsi and time
                        # range is in KM, 1 KM = 1093 YDS = 0.539 NM
                        # require that reported speed is greater than 1 to leave out idling ships, not moving through the water
                        if cpa_rng_min <= rng <= cpa_rng_max and spd >= 1.0:  
                            ship_data = mmsiDB.loc[mmsiDB['MMSI'] == mmsi] 
                            if ship_data.empty:
                                print("WARNING: " + str(mmsi) + " not in DB!")
                                continue
                            # get the values from the Series in ship_data
                            ship_size = ship_data["DWT"].values[0]
                            if ship_size == '-':
                                ship_size = '-1'
                            ship_desig = ship_data["DESIG"].values[0]
                            ##############################
                            # This section can be modified and other label info saved
                            # instead of 'largeShip', 'smallShip'
                            ##############################
                            if(int(ship_size) > 5000):
                                ship_type = "largeShip"
                            elif(int(ship_size) != -1):
                                ship_type = "smallShip"
                            else:
                                ship_type = "UNK"
                            ########################## 
                            
                            # save the data, append if key already in dict
                            if mmsi in daily_ships:
                                # record CPA time, and range
                                if rng < daily_ships[mmsi][5]:
                                    cpa_time = record_time
                                    # if mmsi already in then overwrite to add end_time
                                    start_time = daily_ships[mmsi][1] 
                                    daily_ships[mmsi][2] = record_time
                                    daily_ships[mmsi][5]  = rng
                                    daily_ships[mmsi][6] =  cpa_time
                                    if brg_and_rng:
                                        daily_ships[mmsi][7].append((record_time,brg, rng))
                                else:
                                    # stop updating CPA info
                                    daily_ships[mmsi][2] = record_time
                                    if brg_and_rng:
                                        daily_ships[mmsi][7].append((record_time,brg, rng))

                            elif ship_desig not in ["Unknown", "-1", "Unknown (HAZ-A)", "Other type"]:  
                                # else add new key to dict with only start time, add end time later
                                # don't add unknow ships
                                daily_ships[mmsi] = [mmsi, record_time,0, ship_type, ship_desig, rng, record_time]        
                                if brg_and_rng:
                                    daily_ships[mmsi].append([(record_time, brg, rng)])

                        if cpa_rng_min <= rng <= cpa_rng_beyond and spd >= 0.1:
                            find_no_ships[record_time] = (mmsi, rng, spd)
                    
            h5file.close()
        # remove any 0s in END_TIME
        tmp_ship_df = pd.DataFrame.from_dict(daily_ships, orient='index', columns= column_name)
        time_order_no_ships = []
        
        # time order
        for key in sorted(find_no_ships):
            time_order_no_ships.append([key, find_no_ships[key]])
        # find time periods where the time between 2 ships being within cpa_rng_beyond is at least 1 hour
        for idx in range(0, len(time_order_no_ships)-1):
            if  (time_order_no_ships[idx+1][0] - time_order_no_ships[idx][0]) >= dt.timedelta(hours=1) and time_order_no_ships[idx][1][0] != time_order_no_ships[idx+1][1][0]:
                # append this time segment to the tmp dict
                if brg_and_rng:
                    tmp_ship_df = tmp_ship_df.append({"MMSI":0, "START_TIME":time_order_no_ships[idx][0], "END_TIME":time_order_no_ships[idx+1][0], "LABEL":'not ship', "DESIG":'not ship', "CPA":0, "CPA_TIME":0, "BEARING_RANGE_DATA":0}, ignore_index=True)
                else:
                    tmp_ship_df = tmp_ship_df.append({"MMSI":0, "START_TIME":time_order_no_ships[idx][0], "END_TIME":time_order_no_ships[idx+1][0], "LABEL":'not ship', "DESIG":'not ship', "CPA":0, "CPA_TIME":0}, ignore_index=True)

        # remove any 0s in END_TIME
        tmp_ship_df = tmp_ship_df[tmp_ship_df["END_TIME"] != 0] 

        all_ships = pd.concat([all_ships,tmp_ship_df])
        
        # reset daliy dict
        daily_ships.clear()
        find_no_ships.clear()
    
    # check to make sure that the times are on the same day
    # some AIS files start recording at 23:58 the day before, this is a problem as ship times are assumed to not cross days,
    # correct this by setting start times to 00:00:05
    for ship_sample in all_ships.iterrows():
        if ship_sample[1][1].day < ship_sample[1][2].day:
            ship_sample[1][1] = dt.datetime(ship_sample[1][2].year, ship_sample[1][2].month, ship_sample[1][2].day, 0,0,5)
        
        elif ship_sample[1][1].day > ship_sample[1][2].day:
            ship_sample[1][2] = dt.datetime(ship_sample[1][1].year, ship_sample[1][1].month, ship_sample[1][1].day, 23,59,55)
    
    # set column names
    #all_ships.columns = column_name
    # save a csv so that the user can see what was saved
    all_ships.to_csv(output_filename,index=False, header=True)

    # save a pickle file to make read-in easier, since the pickle file is large, this is only necessary if we want to capture bearing and range data
    if brg_and_rng:
        pickle_save_name = output_filename.split('.')[0] + '.pkl'
        pd.to_pickle(all_ships, pickle_save_name)

def generate_multi_label_ships(data_dir, mmsiDB_file, output_filename, far_rng, max_rng, min_rng=0.0):
    """
    This function reads through all .mat AIS files in the data_dir and provides a single output
    It also uses data from the mmsiDB file to get ship type infomation
    #Arguments:
        data_dir:        path to AIS .mat files to process
        mmsiDB_file:     path to csv file contatining mmsi and ship info
        output_filename: path and name for output csv
        far_rng:         cpa range beyond which no ship is considered present in the data
        max_rng:         maximum cpa range in kilometers from sensor, usually set by user as the upper bound
        min_rng:         minimum cpa range in kilometers from sensor, usually 0
    """
    cwd = os.getcwd() 
    mmsiDB = pd.read_csv(os.path.join(cwd, mmsiDB_file), header=0)
    # remove microsecond data from dataframe
    all_ships = pd.DataFrame(columns=["START_TIME", "END_TIME", "DESIG"])
    # list of ships currently overlapping
    open_ships =[]
    find_no_ships = {}
    for f in glob.glob(data_dir+'*.mat'):
        # store in key = time stamp, value = list of MMSI, range, brg
        with warnings.catch_warnings():  # this suppresses an HDFS datatype warning that is from a particular column in all .mat files
            warnings.simplefilter("ignore")
            h5file = tables.open_file(f, mode = 'r')
            prev_write_time = 0
            saveDict = {}
            sortedDict = {}
            # data is stored by mmsi, in order to find overlapping ships, this needs
            # to be converted to time ordering
            for i, ship in enumerate(h5file.root):
                if i > 0:
                    mmsi = int(ship.MMSI[0][0])
                    for x, time in enumerate(ship.datenumber):
                        t = convertTime(time[0])
                        rng = ship.range[x][0]
                        spd = int(ship.SOG[x][0])
                        # add mmsi and time
                        if t in saveDict and ((rng <= far_rng) and spd >= 1.0):
                            # get the list of MMSIs and update with mmsi
                            saveDict[t].append([mmsi, rng])
                        elif ((rng <= far_rng) and spd >= 1.0):
                            saveDict[t] = [[mmsi, rng]]
            
            h5file.close()         
        
        # sort all AIS data by time
        for element in sorted(saveDict):
            sortedDict[element] = saveDict[element]
            
        for timestep in sortedDict:
            for line in sortedDict[timestep]:
                rng = line[1]
                mmsi = line[0] 
                ship_data = mmsiDB.loc[mmsiDB['MMSI'] == mmsi] 
                if ship_data.empty:
                    print("WARNING: " + str(mmsi) + " not in DB!")
                    continue
                desig = ship_data["DESIG"].values[0]
                # don't add unknow ships to the list
                if desig != '-1' and desig != 'Unknown':
                    # add mmsi and time
                    # range is in KM, 1 KM = 1093 YDS = 0.539 NM
                    # require that reported speed is greater than 1 to leave out idling ships, not moving through the water
                    if min_rng <= rng <= max_rng:  
                        if mmsi not in open_ships and len(open_ships) > 0:
                            # ship is within correct range and not already in open_ships, write open_ships then append new ship
                            desigs = []
                            for s in open_ships:
                                desig = mmsiDB.loc[mmsiDB['MMSI'] == s]["DESIG"].values[0]
                                desigs.append(desig)
                            all_ships = all_ships.append({"START_TIME":prev_write_time ,  "END_TIME": timestep, "DESIG":desigs}, ignore_index=True)
                            open_ships.append(mmsi)
                            # set prev_write_time to now
                            prev_write_time = timestep
                        # case for first ship
                        elif mmsi not in open_ships:
                            prev_write_time = timestep
                            open_ships.append(mmsi)
                    # ship now out of range
                    elif rng > max_rng and mmsi in open_ships:
                        # write out and remove
                        desigs = []
                        for s in open_ships:
                            desig = mmsiDB.loc[mmsiDB['MMSI'] == s]["DESIG"].values[0]
                            desigs.append(desig)
                        all_ships = all_ships.append({"START_TIME":prev_write_time ,  "END_TIME": timestep, "DESIG": desigs}, ignore_index=True)
                        open_ships.remove(mmsi)
                        # set prev_write_time to now
                        prev_write_time = timestep
                    # ship now out of range
                    elif rng < min_rng and mmsi in open_ships:
                        # write out and remove
                        desigs = []
                        for s in open_ships:
                            desig = mmsiDB.loc[mmsiDB['MMSI'] == s]["DESIG"].values[0]
                            desigs.append(desig)
                        all_ships = all_ships.append({"START_TIME":prev_write_time ,  "END_TIME": timestep, "DESIG": desigs}, ignore_index=True)
                        open_ships.remove(mmsi)
                        # set prev_write_time to now
                        prev_write_time = timestep
                    
                    if min_rng <= rng <= far_rng and spd >= 0.1:
                        find_no_ships[timestep] = (mmsi, rng, spd)
        
        time_order_no_ships = [] 
        for key in sorted(find_no_ships):
            time_order_no_ships.append([key, find_no_ships[key]])
        
        # find time periods where the time between 2 ships being within cpa_rng_beyond is at least 1 hour
        for idx in range(0, len(time_order_no_ships)-1):
            if  (time_order_no_ships[idx+1][0] - time_order_no_ships[idx][0]) >= dt.timedelta(hours=1) and time_order_no_ships[idx][1][0] != time_order_no_ships[idx+1][1][0]:
                # append this time segment to the tmp dict
                all_ships = all_ships.append({"START_TIME":time_order_no_ships[idx][0], "END_TIME":time_order_no_ships[idx+1][0], "DESIG":'not ship'}, ignore_index=True)

        # reset daliy dict
        open_ships.clear()
        find_no_ships.clear()
    
    # check to make sure that the times are on the same day
    # some AIS files start recording at 23:58 the day before, this is a problem as ship times are assumed to not cross days,
    # correct this by setting start times to 00:00:05
    for ship_sample in all_ships.iterrows():
        if ship_sample[1][0].day < ship_sample[1][1].day:
            ship_sample[1][0] = dt.datetime(ship_sample[1][1].year, ship_sample[1][1].month, ship_sample[1][1].day, 0,0,5)
        
        elif ship_sample[1][0].day > ship_sample[1][1].day:
            ship_sample[1][1] = dt.datetime(ship_sample[1][0].year, ship_sample[1][0].month, ship_sample[1][0].day, 23,59,55)

    all_ships.to_csv(output_filename,index=False, header=True)


def gen_data_per_file(dir, cpa_rng_max, cpa_rng_min):
    """
        The function generates one output file for each .mat file in a directory. As opposed to generate_single_output that creates one file for all .mat files
        #Arguments:
            dir:         path to AIS .mat files to process
            cpa_rng_max: maximum cpa range in kilometers from sensor, usually set by user as the upper bound
            cpa_rng_min: minimum cpa range in kilometers from sensor, usually 0
    """
    cwd = os.getcwd()
    # collect all mmsi data to search the web for these mmsi to get ship data
    all_mmsis = {}
    for filename in glob.glob(dir+'*.mat'):
        fName = os.path.splitext(filename)[0]
        # store in key = time stamp, value = list of MMSI, range, brg
        h5file = tables.open_file(filename, mode = 'r')
        saveDict = {}
        for i, ship in enumerate(h5file.root):
            imo = 0
            sType = 0
            if i > 0:
                mmsi = int(ship.MMSI[0][0])
                if 'IMOnumber' in ship:
                    imo = int(ship.IMOnumber[0][0])
                if 'ShipType' in ship:
                    sType = int(ship.ShipType[0][0])
                # if mmsi not found yet, add to list
                if mmsi not in all_mmsis:
                    all_mmsis[mmsi] = [mmsi, imo,sType]
                for x, time in enumerate(ship.datenumber):
                    t = convertTime(time[0])
                    rng = ship.range[x][0]
                    brg = ship.bearing[x][0]
                    spd = int(ship.SOG[x][0])
                    # add mmsi and time
                    if t in saveDict and ((cpa_rng_min <= rng <= cpa_rng_max) and spd >= 1.0):
                        # get the list of MMSIs and update with mmsi
                        saveDict[t].append([mmsi, rng, brg, spd])
                    elif ((cpa_rng_min <= rng <= cpa_rng_max) and spd >= 1.0):
                        saveDict[t] = [[mmsi, rng, brg, spd ]]
        
        h5file.close()
        # convert to dataframe to write out to csv
        # order dict by key, super simple method
        sortedDict = {}
        for element in sorted(saveDict):
            sortedDict[element] = saveDict[element]

    # save all mmsis found for web search to build mmsiDB
    pd.DataFrame.from_dict(all_mmsis, orient= 'index' ,columns=["MMSI", "IMO", "Ship Type"]).to_csv("all_mmsis.csv", index=False, header=True)

def find_mmsi(tgt_dir, tgt_mmsi, output_csv):
    """
        This function gets only the data for mmsi from the target file. Useful when plotting results, in order to get all of the data for 1 or more ships on a specific day to
        plot.
    
        #Arguments:
             tgt_mmsi: a list of 1 or more target mmsis to find
    """
    cwd = os.getcwd()
    saveDict = {}
    print(tgt_mmsi)
    for filename in glob.glob(tgt_dir+'*.mat'):
        # store in key = time stamp, value = list of MMSI, range, brg
        h5file = tables.open_file(filename, mode = 'r')
        for i, ship in enumerate(h5file.root):
            if i > 0:
                mmsi = int(ship.MMSI[0][0])
                if mmsi in tgt_mmsi:
                    for x, time in enumerate(ship.datenumber):
                        t = convertTime(time[0])
                        rng = ship.range[x][0]
                        brg = ship.bearing[x][0]
                        # add mmsi to save dict
                        saveDict[t] = [mmsi, rng, brg]
        
        h5file.close()
    # convert to dataframe to write out to csv
    # order dict by key, super simple method, this puts all data in correct time order
    sortedDict = {}
    for element in sorted(saveDict):
        sortedDict[element] = saveDict[element]
    save_f_name = os.path.join(cwd, output_csv)
    pd.DataFrame.from_dict(sortedDict, orient = 'index').to_csv(save_f_name, index_label='TIME', header=['MMSI', 'RNG', 'BRG'])

def main():
    """
        Collect and process command line arguments
    """
    str_to_bool = lambda x : True if x.lower() == 'true' else False
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--target_dir',    type=str,   help="Target directory of .mat files containing AIS data. Or single mat file in mmsi mode")
    parser.add_argument('--mmsiDB_file',   type=str,   help="Name of csv file containing MMSI data from web.")
    parser.add_argument('--cpa_max_range', type=float, help="Max CPA range in KM to return ship data for")
    parser.add_argument('--cpa_min_range', type=float, help="Min CPA range in KM to return ship data for. Default to 0",        default=0.0)
    parser.add_argument('--cpa_no_range',  type=float, help="Range beyond which data is considered 'not ship. Default is 30.0", default=30.0)
    parser.add_argument('--mode',          type=str,   help="Function to run, either single or many. Single generates one output csv file for all AIS .mat files. Many generates a csv file for each .mat file, Many also creates list of MMSIs for web search")
    parser.add_argument('--mmsi',          type=str,   help="Target mmsi to search for when in mmsi mode.", default='1')
    parser.add_argument('--output_csv',    type=str,   help="Name of output csv file for single mode.")
    
    parser.add_argument('--brg_and_rng',   type=str_to_bool,   help="If True, collect bearing and range data, default is false",  default='False')

    args = parser.parse_args()

    # parse comma sep list of mmsis into list
    mmsis = [int(x) for x in args.mmsi.split(',')]

    if args.mode == 'single':
        generate_single_output(args.target_dir, args.mmsiDB_file, args.output_csv, args.brg_and_rng, args.cpa_no_range, args.cpa_max_range, args.cpa_min_range)
    elif args.mode == 'multilabel':
        generate_multi_label_ships(args.target_dir, args.mmsiDB_file, args.output_csv, args.cpa_no_range, args.cpa_max_range, args.cpa_min_range)
    elif args.mode == 'many':
        gen_data_per_file(args.target_dir, args.cpa_max_range, args.cpa_min_range)
    elif args.mode == 'mmsi':
        find_mmsi(args.target_dir, mmsis, args.output_csv)
    else:
        sys.exit("WARNING: Mode not supported!")

if __name__ == "__main__":
    main()
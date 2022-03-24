import datetime as dt
import tables
import pandas as pd
import os
import sys
import glob
from bs4 import BeautifulSoup
import requests
import argparse
from pyproj import Proj, transform, Transformer
from geographiclib.geodesic import Geodesic
import geopy.distance
import warnings
#-------------------update variables to run script--------------------------
# filename is the .mat of the ais tracks and data
# mmsi_list are the mmsis of the ships whos positions to scrape from the .mat file
# save_csv is the name and filepath of the output csv to save the dataframe to
# Script must be adapted if you wish to run against multiple .mat files
#filename = '/h/nicholas.villemez/thesis/acoustic-inference-application/data/for_visualization/harp_ship_data_2/ais/130225.mat'
#mmsi_list = [373817000]
#save_csv = '/h/nicholas.villemez/thesis/acoustic-inference-application/data/for_visualization/harp_ship_data_2/viz_data/harp_single_ais_2.csv'
#---------------------------------------------------------------------------

def get_bearing(to_lat, to_long):
    # Sensor location
    from_lat = 36.712465
    from_long = -122.187548
    bearing = Geodesic.WGS84.Inverse(from_lat, from_long, to_lat, to_long)['azi1'] % 360
    return bearing

# First set of latitude and longitude is the center point, second set is being compared
def compare_lat_long(lat2, lon2):
    """
    Calculate the distance in kilometers between two points 
    on the earth (specified in decimal degrees, uses WGS-84 ellipsoidal distance)
    """
    # Sensor location
    lat1 = 36.712465
    lon1 = -122.187548
    
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    distance = geopy.distance.geodesic(coords_1, coords_2).km # use .miles for miles
    return distance

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

def scrapeDesig(mmsi_list, imos):
    base_html = "https://www.vesselfinder.com/vessels?name="
    # needed to fool server into thinking we're a browser
    headers = headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}
    mmsi_db = pd.DataFrame(columns=["MMSI","DESIG"]) 

    #for mmsi_data in mmsis.iterrows():
    for i in range(0, len(mmsi_list)):
        tgt = base_html + str(mmsi_list[i])
        # get the html and pass into the soup parser
        html = requests.get(tgt, headers=headers)
        # verify we get the page ok
        if html.status_code == 200:
            html_dom = BeautifulSoup(html.content, features="html.parser")
            # verfiy we did not get 'No Results'
            check = html_dom.find("section", attrs={'class',"listing"}).get_text()
            if check != "No resultsRefine your criteria and search again":
                try:
                    des = html_dom.find("td", attrs={'class': 'v2'}).find("div", attrs={'class': 'slty'}).get_text()
                except:
                    print("issue with parser!", flush=True)
                    des = -1
                mmsi_db = mmsi_db.append({'MMSI':mmsi_list[i], 'DESIG':des}, ignore_index=True)

            # try again with IMO number if we have one
            elif imos[i] > 0:
                tgt = base_html + str(imos[i])
                # get the html and pass into the soup parser
                html = requests.get(tgt, headers=headers)
                # verify we get the page ok
                if html.status_code == 200:
                    html_dom = BeautifulSoup(html.content, features="html.parser")
                    # verfiy we did not get 'No Results'
                    check = html_dom.find("section", attrs={'class',"listing"}).get_text()
                    if check != "No resultsRefine your criteria and search again":
                        try:
                            des = html_dom.find("td", attrs={'class': 'v2'}).find("div",attrs={'class': 'slty'}).get_text()
                        except:
                            print("issue with parser!", flush=True)
                            des = -1
                        mmsi_db = mmsi_db.append({'MMSI':mmsi_list[i], 'DESIG':des}, ignore_index=True)
                    else:
                        mmsi_db = mmsi_db.append({'MMSI':mmsi_list[i], 'DESIG':-1}, ignore_index=True)
                else:
                    mmsi_db = mmsi_db.append({'MMSI':mmsi_list[i], 'DESIG':-1}, ignore_index=True)

            else:
                print("Did not find MMSI and IMO: " + str(mmsi_list[i]) + " " + str(imos[i]), flush=True)
                # still append to list so we don't search again, all values -1 as flag
                mmsi_db = mmsi_db.append({'MMSI':mmsi_list[i], 'DESIG':-1}, ignore_index=True)
        else:
            print("Failed to retrieve page " + str(tgt), flush=True)
            mmsi_db = mmsi_db.append({'MMSI':mmsi_list[i], 'DESIG':-1}, ignore_index=True)

    # all done, write out mmsi_db
    return mmsi_db
'''
h5file = tables.open_file(filename, mode = 'r')
saveDict = {}
for i, ship in enumerate(h5file.root):
    if i > 0:
            mmsi = int(ship.MMSI[0][0])
            if mmsi in mmsi_list:
                    for x, time in enumerate(ship.datenumber):
                            t = convertTime(time[0])
                            rng = ship.range[x][0]
                            brg = ship.bearing[x][0]
                            lat = ship.LAT[x][0]
                            long = ship.LON[x][0]
                            #desig = ship.ShipType[0][0]
                            imo = ship.IMOnumber[0][0]
                            saveDict[t] = [mmsi, rng, brg, lat, long, imo]

            
sortedDict = {}
for element in sorted(saveDict):
    sortedDict[element] = saveDict[element]

ais_track_df = pd.DataFrame.from_dict(sortedDict, orient = 'index')

ais_track_df.columns=['MMSI', 'RNG', 'BRG', 'LAT', 'LONG', 'IMO']

imos=[]
for x in mmsi_list:
    imos.append(ais_track_df.loc[ais_track_df['MMSI'] == x, 'IMO'].iloc[0])

desig_df = scrapeDesig(mmsi_list, imos)

ais_track_df = ais_track_df.reset_index().merge(desig_df, how='left', on='MMSI').set_index('index')

ship_class_dict ={'Landings Craft':'classA', 'Military ops':'classA','Fishing vessel':'classA','Fishing Vessel':'classA' ,'Fishing Support Vessel':'classA', 'Tug':'classA', 'Pusher Tug':'classA', 'Dredging or UW ops':'classA', 'Towing vessel':'classA', 'Crew Boat':'classA', 'Buoy/Lighthouse Vessel':'classA', 'Salvage Ship':'classA', 'Research Vessel':'classA', 'Anti-polution':'classA', 'Offshore Tug/Supply Ship':'classA', 'Law enforcment':'classA', 'Landing Craft':'classA', 'SAR':'classA', 'Patrol Vessel':'classA', 'Pollution Control Vessel': 'classA', 'Offshore Support Vessel':'classA',
                        'Pleasure craft':'classB', 'Yacht':'classB', 'Sailing vessel':'classB', 'Pilot':'classB', 'Diving ops':'classB', 
                        'Passenger (Cruise) Ship':'classC', 'Passenger Ship':'classC', 'Passenger ship':'classC', 'Training Ship': 'classC',
                        'Naval/Naval Auxiliary':'classD','DDG':'classD','LCS':'classD','Hospital Vessel':'classD' ,'Self Discharging Bulk Carrier':'classD' ,'Cutter':'classD', 'Passenger/Ro-Ro Cargo Ship':'classD', 'Heavy Load Carrier':'classD', 'Vessel (function unknown)':'classD',
                        'General Cargo Ship':'classD','Wood Chips Carrier':'classD', 'Bulk Carrier':'classD' ,'Cement Carrier':'classD','Vehicles Carrier':'classD','Cargo ship':'classD', 'Oil Products Tanker':'classD', 'Ro-Ro Cargo Ship':'classD', 'USNS RAINIER':'classD', 'Supply Tender':'classD', 'Cargo ship':'classD', 'LPG Tanker':'classD', 'Crude Oil Tanker':'classD', 'Container Ship':'classD', 'Container ship':'classD','Bulk Carrier':'classD', 'Chemical/Oil Products Tanker':'classD', 'Refrigerated Cargo Ship':'classD', 'Tanker':'classD', 'Car Carrier':'classD', 'Deck Cargo Ship' :'classD', 'Livestock Carrier': 'classD',
                        'Bunkering Tanker':'classD', 'Water Tanker': 'classD', 'FSO': 'classD', 
                        'not ship':'classE', -1:'Unknown' }

ais_track_df['SHIP_CLASS'] = ais_track_df['DESIG'].map(ship_class_dict)

ais_track_df = ais_track_df.reset_index().rename(columns={'index':'TIME'})

ais_track_df.to_csv(save_csv, index=False)
'''




# Original functions from Andrew's code

# Creates dataframe for mmsidb
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
    for filename in dir: #glob.glob(dir+'/*.mat'):
        with warnings.catch_warnings():  # this suppresses an HDFS datatype warning that is from a particular column in all .mat files
            warnings.simplefilter("ignore")
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
                    # May not need this part
                    '''
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
                    '''
            
            h5file.close()
            # convert to dataframe to write out to csv
            # order dict by key, super simple method
            '''
            sortedDict = {}
            for element in sorted(saveDict):
                sortedDict[element] = saveDict[element]
            '''

    # save all mmsis found for web search to build mmsiDB
    all_mmsis = pd.DataFrame.from_dict(all_mmsis, orient= 'index' ,columns=["MMSI", "IMO", "Ship Type"]) #.to_csv("all_mmsis.csv", index=False, header=True)
    all_mmsis = all_mmsis.reset_index().drop('index', axis=1)
    
    return all_mmsis



# Then use mmsi scraper to get information from internet
'''
 Andrew Pfau
 script to auto scrape ship data from internet
 site: www.vesselfinder.com/vessels?name= MMSI Number
 Data recorded is MMSI, Dead Weigth Tonnage, description, and Size in meters (length / beam)
 Not every vessel has all this data, smaller ships do not lsit DWT

 DWT is a measure of weight a ship can carry, site does not list ship tonnage 

 Arguments: mmsi csv, a csv file with all of the mmsis to retrieve information for
            mmsiDB csv file name of where to save the data

'''

from bs4 import BeautifulSoup
import sys
import os
import pandas as pd
import requests
import glob
import argparse

def mmsi_scraper(mmsis):
    # read input file of new mmsis to search for
    #pd.read_csv(os.path.join(dir, mmsiFile), header=0, names=["MMSI","IMO", "TYPE"] )
    
    # dataframe to save ship data from web in, will become output file later
    mmsi_db = pd.DataFrame(columns=["MMSI","DWT","SIZE","DESIG"]) 

    base_html = "https://www.vesselfinder.com/vessels?name="
    # needed to fool server into thinking we're a browser
    headers = headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}
    
    for mmsi_data in mmsis.iterrows():
        tgt = base_html + str(mmsi_data[1]['MMSI'])
        # get the html and pass into the soup parser
        html = requests.get(tgt, headers=headers)
        # verify we get the page ok
        if html.status_code == 200:
            html_dom = BeautifulSoup(html.content, features="html.parser")
            # verfiy we did not get 'No Results'
            check = html_dom.find("section", attrs={'class',"listing"}).get_text()
            if check != "No resultsRefine your criteria and search again":
                # dead weight tonnage
                dwt = html_dom.find("td", attrs={'class':'v5 is-hidden-mobile'}).get_text()

                if dwt == '-':
                    dwt=-1
                # size in meters, length / beam
                size = html_dom.find("td", attrs={'class':'v6 is-hidden-mobile'}).get_text()
                # ship description
                #des = html_dom.find("td", attrs={'class':'v2'}).find('small').get_text()
                try:
                    des = html_dom.find("td", attrs={'class': 'v2'}).find("div", attrs={'class': 'slty'}).get_text()
                except:
                    print("issue with parser!", flush=True)
                    des = 'Unknown'
                mmsi_db = mmsi_db.append({'MMSI':mmsi_data[1]['MMSI'], 'DWT':dwt, 'SIZE':size,'DESIG':des}, ignore_index=True)

            # try again with IMO number if we have one
            elif mmsi_data[1][1] > 0:
                tgt = base_html + str(mmsi_data[1]['IMO'])
                # get the html and pass into the soup parser
                html = requests.get(tgt, headers=headers)
                # verify we get the page ok
                if html.status_code == 200:
                    html_dom = BeautifulSoup(html.content, features="html.parser")
                    # verfiy we did not get 'No Results'
                    check = html_dom.find("section", attrs={'class',"listing"}).get_text()
                    if check != "No resultsRefine your criteria and search again":
                        # dead weight tonnage
                        dwt = html_dom.find("td", attrs={'class':'v5 is-hidden-mobile'}).get_text()
                        if dwt == '-':
                            dwt=-1
                        # size in meters, length / beam
                        size = html_dom.find("td", attrs={'class':'v6 is-hidden-mobile'}).get_text()
                        try:
                            des = html_dom.find("td", attrs={'class': 'v2'}).find("div",attrs={'class': 'slty'}).get_text()
                        except:
                            print("issue with parser!", flush=True)
                            des = 'Unknown'
                        mmsi_db = mmsi_db.append({'MMSI':mmsi_data[1]['MMSI'], 'DWT':dwt, 'SIZE':size,'DESIG':des}, ignore_index=True)

                    else:
                        print("Did not find MMSI and IMO: " + str(mmsi_data[1]['MMSI']) + " " + str(mmsi_data[1]['IMO']), flush=True)
                        mmsi_db = mmsi_db.append({'MMSI':mmsi_data[1]['MMSI'], 'DWT':-1, 'SIZE':-1, 'DESIG':-1}, ignore_index=True)
                
                else:
                    print("Did not find MMSI and IMO: " + str(mmsi_data[1]['MMSI']) + " " + str(mmsi_data[1]['IMO']), flush=True)
                    mmsi_db = mmsi_db.append({'MMSI':mmsi_data[1]['MMSI'], 'DWT':-1, 'SIZE':-1, 'DESIG':-1}, ignore_index=True)

            else:
                print("Did not find MMSI and IMO: " + str(mmsi_data[1]['MMSI']) + " " + str(mmsi_data[1]['IMO']))
                # still append to list so we don't search again, all values -1 as flag
                mmsi_db = mmsi_db.append({'MMSI':mmsi_data[1]['MMSI'], 'DWT':-1, 'SIZE':-1, 'DESIG':-1}, ignore_index=True)
        else:
            print("Failed to retrieve page " + str(tgt), flush=True)
            mmsi_db = mmsi_db.append({'MMSI':mmsi_data[1]['MMSI'], 'DWT':-1, 'SIZE':-1, 'DESIG':-1}, ignore_index=True)

    # all done, write out mmsi_db
    #mmsi_db.to_csv(os.path.join(cwd,mmsiDB), index=False)

    return mmsi_db

# Probably don't need to use this, but it does what the above one does
def generate_single_output(data_dir, mmsiDB, cpa_rng_beyond, cpa_rng_max, position_range, cpa_rng_min=0.0):
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
    #cwd = os.getcwd() 
    #mmsiDB = pd.read_csv(os.path.join(cwd, mmsiDB_file), header=0)
    all_ships = pd.DataFrame()
    all_ships_beyond = pd.DataFrame()
    position_ships = pd.DataFrame()
    # format is key = mmsi, values = start_time, end_time, label, desig
    daily_ships ={} 
    find_no_ships = {}
    daily_ships_beyond = {}
    daily_position_ships = {}
    
    #column_name = ["MMSI","START_TIME", "END_TIME", "LABEL", "DESIG", "CPA", "CPA_TIME" ]
    
    column_name = ["MMSI","START_TIME", "END_TIME", "LABEL", "DESIG", "CPA", "CPA_TIME", "MMSI,TIME,LAT,LON,BRG,RNG,DESIG"]
    
    for f in data_dir: #glob.glob(data_dir+'/*.mat'):
        # store in key = time stamp, value = list of MMSI, range, brg
        with warnings.catch_warnings():  # this suppresses an HDFS datatype warning that is from a particular column in all .mat files
            warnings.simplefilter("ignore")
            h5file = tables.open_file(f, mode = 'r')
            for i, ship in enumerate(h5file.root):
                if i > 0:
                    mmsi = int(ship.MMSI[0][0])
                    for x, time in enumerate(ship.datenumber):
                        record_time = convertTime(time[0])

                        lat = ship.LAT[x][0]
                        lon = ship.LON[x][0]

                        brg = get_bearing(lat, lon)

                        rng = compare_lat_long(lat, lon)

                        # Need to change this to custom bearing and range functions
                        #rng = round(ship.range[x][0], 3)
                        spd = int(ship.SOG[x][0])
                        #brg = round(ship.bearing[x][0], 3)
                        # add mmsi and time
                        # range is in KM, 1 KM = 1093 YDS = 0.539 NM
                        # require that reported speed is greater than 1 to leave out idling ships, not moving through the water
                        if cpa_rng_min <= rng <= cpa_rng_max and spd >= 1.0:  
                            ship_data = mmsiDB.loc[mmsiDB['MMSI'] == mmsi] 
                            if ship_data.empty:
                                print("WARNING: " + str(mmsi) + " not in DB!", flush=True)
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
                                    #if brg_and_rng:
                                    daily_ships[mmsi][7].append([str(mmsi), record_time.strftime("%m/%d/%Y-%H:%M:%S"),str(lat),str(lon),str(brg),str(rng),str(ship_desig)])
                                else:
                                    # stop updating CPA info
                                    daily_ships[mmsi][2] = record_time
                                    #if brg_and_rng:
                                    daily_ships[mmsi][7].append([str(mmsi), record_time.strftime("%m/%d/%Y-%H:%M:%S"),str(lat),str(lon),str(brg),str(rng),str(ship_desig)])

                            elif ship_desig not in ["Unknown", "-1", "Unknown (HAZ-A)", "Other type"]:  
                                # else add new key to dict with only start time, add end time later
                                # don't add unknown ships
                                daily_ships[mmsi] = [str(mmsi), record_time,record_time, str(ship_type), str(ship_desig), rng, record_time, [[str(mmsi), record_time.strftime("%m/%d/%Y-%H:%M:%S"),str(lat),str(lon),str(brg),str(rng),str(ship_desig)]] ]        
                                #if brg_and_rng:
                                #daily_ships[mmsi].append([[record_time, brg, rng]])

                        # Make the same information but for cpa range beyond
                        if cpa_rng_min <= rng <= cpa_rng_beyond and spd >= 1.0:  
                            ship_data = mmsiDB.loc[mmsiDB['MMSI'] == mmsi] 
                            if ship_data.empty:
                                print("WARNING: " + str(mmsi) + " not in DB!", flush=True)
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
                            if mmsi in daily_ships_beyond:
                                # record CPA time, and range
                                if rng < daily_ships_beyond[mmsi][5]:
                                    cpa_time = record_time
                                    # if mmsi already in then overwrite to add end_time
                                    start_time = daily_ships_beyond[mmsi][1] 
                                    daily_ships_beyond[mmsi][2] = record_time
                                    daily_ships_beyond[mmsi][5]  = rng
                                    daily_ships_beyond[mmsi][6] =  cpa_time
                                    #if brg_and_rng:
                                    daily_ships_beyond[mmsi][7].append([str(mmsi), record_time.strftime("%m/%d/%Y-%H:%M:%S"),str(lat),str(lon),str(brg),str(rng),str(ship_desig)])
                                else:
                                    # stop updating CPA info
                                    daily_ships_beyond[mmsi][2] = record_time
                                    #if brg_and_rng:
                                    daily_ships_beyond[mmsi][7].append([str(mmsi), record_time.strftime("%m/%d/%Y-%H:%M:%S"),str(lat),str(lon),str(brg),str(rng),str(ship_desig)])

                            elif ship_desig not in ["Unknown", "-1", "Unknown (HAZ-A)", "Other type"]:  
                                # else add new key to dict with only start time, add end time later
                                # don't add unknown ships
                                daily_ships_beyond[mmsi] = [str(mmsi), record_time,record_time, str(ship_type), str(ship_desig), rng, record_time, [[str(mmsi), record_time.strftime("%m/%d/%Y-%H:%M:%S"),str(lat),str(lon),str(brg),str(rng),str(ship_desig)]] ]        
                                #if brg_and_rng:
                                #daily_ships[mmsi].append([[record_time, brg, rng]])

                        # Make the same information but for position range
                        #if cpa_rng_min <= rng <= position_range:  
                        ship_data = mmsiDB.loc[mmsiDB['MMSI'] == mmsi] 
                        if ship_data.empty:
                            print("WARNING: " + str(mmsi) + " not in DB!", flush=True)
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
                        if mmsi in daily_position_ships:
                            # record CPA time, and range
                            if rng < daily_position_ships[mmsi][5]:
                                cpa_time = record_time
                                # if mmsi already in then overwrite to add end_time
                                start_time = daily_position_ships[mmsi][1] 
                                daily_position_ships[mmsi][2] = record_time
                                daily_position_ships[mmsi][5]  = rng
                                daily_position_ships[mmsi][6] =  cpa_time
                                #if brg_and_rng:
                                daily_position_ships[mmsi][7].append([str(mmsi), record_time.strftime("%m/%d/%Y-%H:%M:%S"),str(lat),str(lon),str(brg),str(rng),str(ship_desig)])
                            else:
                                # stop updating CPA info
                                daily_position_ships[mmsi][2] = record_time
                                #if brg_and_rng:
                                daily_position_ships[mmsi][7].append([str(mmsi), record_time.strftime("%m/%d/%Y-%H:%M:%S"),str(lat),str(lon),str(brg),str(rng),str(ship_desig)])

                        elif ship_desig not in ["Unknown", "-1", "Unknown (HAZ-A)", "Other type"]:  
                            # else add new key to dict with only start time, add end time later
                            # don't add unknown ships
                            daily_position_ships[mmsi] = [str(mmsi), record_time,record_time, str(ship_type), str(ship_desig), rng, record_time, [[str(mmsi), record_time.strftime("%m/%d/%Y-%H:%M:%S"),str(lat),str(lon),str(brg),str(rng),str(ship_desig)]] ]        
                            #if brg_and_rng:
                            #daily_ships[mmsi].append([[record_time, brg, rng]])

                        #if cpa_rng_min <= rng <= cpa_rng_beyond and spd >= 0.1:
                        #    find_no_ships[record_time] = (mmsi, rng, spd)
                    
            h5file.close()
        # remove any 0s in END_TIME, shouldn't matter though because all ships are entered with no 0s
        tmp_ship_df = pd.DataFrame.from_dict(daily_ships, orient='index', columns= column_name)
        tmp_ship_beyond_df = pd.DataFrame.from_dict(daily_ships_beyond, orient='index', columns= column_name)
        tmp_position_ships = pd.DataFrame.from_dict(daily_position_ships, orient='index', columns= column_name)
        #time_order_no_ships = []
        
        # time order
        #for key in sorted(find_no_ships):
        #    time_order_no_ships.append([key, find_no_ships[key]])
        # find time periods where the time between 2 ships being within cpa_rng_beyond is at least 1 hour
        '''
        for idx in range(0, len(time_order_no_ships)-1):
            if  (time_order_no_ships[idx+1][0] - time_order_no_ships[idx][0]) >= dt.timedelta(hours=1) and time_order_no_ships[idx][1][0] != time_order_no_ships[idx+1][1][0]:
                # append this time segment to the tmp dict
                #if brg_and_rng:
                tmp_ship_df = tmp_ship_df.append({"MMSI":'0', "START_TIME":time_order_no_ships[idx][0], "END_TIME":time_order_no_ships[idx+1][0], "LABEL":'not ship', "DESIG":'not ship', "CPA":'0', "CPA_TIME":'0', "MMSI,TIME,LAT,LON,BRG,RNG":[['0','0','0','0','0','0']]}, ignore_index=True)
                #else:
                #tmp_ship_df = tmp_ship_df.append({"MMSI":0, "START_TIME":time_order_no_ships[idx][0], "END_TIME":time_order_no_ships[idx+1][0], "LABEL":'not ship', "DESIG":'not ship', "CPA":0, "CPA_TIME":0}, ignore_index=True)
        '''

        # Changed the above to account for all ranges of time between 2 ships being within cpa_rng_beyond
        # And just increased the range for cpa_range_beyond
        


        # remove any 0s in END_TIME
        tmp_ship_df = tmp_ship_df[tmp_ship_df["END_TIME"] != 0] 
        tmp_ship_beyond_df = tmp_ship_beyond_df[tmp_ship_beyond_df["END_TIME"] != 0] 
        tmp_position_ships = tmp_position_ships[tmp_position_ships["END_TIME"] != 0] 

        all_ships = pd.concat([all_ships,tmp_ship_df])
        all_ships_beyond = pd.concat([all_ships_beyond,tmp_ship_beyond_df])
        position_ships = pd.concat([position_ships,tmp_position_ships])
        
        # reset daily dict
        daily_ships.clear()
        find_no_ships.clear()
        daily_ships_beyond.clear()
        daily_position_ships.clear()
    
    # check to make sure that the times are on the same day
    # some AIS files start recording at 23:58 the day before, this is a problem as ship times are assumed to not cross days,
    # correct this by setting start times to 00:00:05
    '''
    for ship_sample in all_ships.iterrows():
        if ship_sample[1][1].day < ship_sample[1][2].day:
            ship_sample[1][1] = dt.datetime(ship_sample[1][2].year, ship_sample[1][2].month, ship_sample[1][2].day, 0,0,5)
        
        elif ship_sample[1][1].day > ship_sample[1][2].day:
            ship_sample[1][2] = dt.datetime(ship_sample[1][1].year, ship_sample[1][1].month, ship_sample[1][1].day, 23,59,55)

    for ship_sample in all_ships_beyond.iterrows():
        if ship_sample[1][1].day < ship_sample[1][2].day:
            ship_sample[1][1] = dt.datetime(ship_sample[1][2].year, ship_sample[1][2].month, ship_sample[1][2].day, 0,0,5)
        
        elif ship_sample[1][1].day > ship_sample[1][2].day:
            ship_sample[1][2] = dt.datetime(ship_sample[1][1].year, ship_sample[1][1].month, ship_sample[1][1].day, 23,59,55)
    '''
    # set column names
    #all_ships.columns = column_name
    # save a csv so that the user can see what was saved
    #all_ships.to_csv(output_filename,index=False, header=True)

    # save a pickle file to make read-in easier, since the pickle file is large, this is only necessary if we want to capture bearing and range data
    '''
    if brg_and_rng:
        pickle_save_name = output_filename.split('.')[0] + '.pkl'
        pd.to_pickle(all_ships, pickle_save_name)
    '''

    all_ships = all_ships.reset_index().drop('index', axis=1)
    all_ships_beyond = all_ships_beyond.reset_index().drop('index', axis=1)
    position_ships = position_ships.reset_index().drop('index', axis=1)
    
    # Use all_ships_beyond to determine the times for which there are "no ships" present
    # Use all_ship to determine the labels for other time periods
    return all_ships, all_ships_beyond, position_ships


# Doesn't work as is, needs modification

'''

# Then create the mmsi list with start and end times

# I'm not sure what the difference between this and single is, but I think this is the one we need to use
def generate_multi_label_ships(data_dir, mmsiDB, far_rng, max_rng, min_rng=0.0):
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
    #cwd = os.getcwd() 
    #mmsiDB = pd.read_csv(os.path.join(cwd, mmsiDB), header=0)
    # remove microsecond data from dataframe
    all_ships = pd.DataFrame(columns=["START_TIME", "END_TIME", "DESIG"])
    # list of ships currently overlapping
    open_ships =[]
    find_no_ships = {}
    for f in glob.glob(data_dir+'/*.mat'):
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

    #all_ships.to_csv(output_filename,index=False, header=True)
    return all_ships
    '''
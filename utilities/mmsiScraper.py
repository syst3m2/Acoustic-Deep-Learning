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

def main(mmsiFile,mmsiDB , dir):
    # read input file of new mmsis to search for
    mmsis = pd.read_csv(os.path.join(dir, mmsiFile), header=0, names=["MMSI","IMO", "TYPE"] )
    
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
            html_dom = BeautifulSoup(html.content)
            # verfiy we did not get 'No Results'
            check = html_dom.find("section", attrs={'class',"listing"}).get_text()
            if check != "No resultsRefine your criteria and search again":
                # dead weight tonnage
                dwt = html_dom.find("td", attrs={'class':'v5 is-hidden-mobile'}).get_text()
                # size in meters, length / beam
                size = html_dom.find("td", attrs={'class':'v6 is-hidden-mobile'}).get_text()
                # ship description
                #des = html_dom.find("td", attrs={'class':'v2'}).find('small').get_text()
                try:
                    des = html_dom.find("td", attrs={'class': 'v2'}).find("div", attrs={'class': 'slty'}).get_text()
                except:
                    print("issue with parser!")
                    des = 'Unknown'
                mmsi_db = mmsi_db.append({'MMSI':mmsi_data[1]['MMSI'], 'DWT':dwt, 'SIZE':size,'DESIG':des}, ignore_index=True)

            # try again with IMO number if we have one
            elif mmsi_data[1][1] > 0:
                tgt = base_html + str(mmsi_data[1]['IMO'])
                # get the html and pass into the soup parser
                html = requests.get(tgt, headers=headers)
                # verify we get the page ok
                if html.status_code == 200:
                    html_dom = BeautifulSoup(html.content)
                    # verfiy we did not get 'No Results'
                    check = html_dom.find("section", attrs={'class',"listing"}).get_text()
                    if check != "No resultsRefine your criteria and search again":
                        # dead weight tonnage
                        dwt = html_dom.find("td", attrs={'class':'v5 is-hidden-mobile'}).get_text()
                        # size in meters, length / beam
                        size = html_dom.find("td", attrs={'class':'v6 is-hidden-mobile'}).get_text()
                        try:
                            des = html_dom.find("td", attrs={'class': 'v2'}).find("div",attrs={'class': 'slty'}).get_text()
                        except:
                            print("issue with parser!")
                            des = 'Unknown'
                        mmsi_db = mmsi_db.append({'MMSI':mmsi_data[1]['MMSI'], 'DWT':dwt, 'SIZE':size,'DESIG':des}, ignore_index=True)

            else:
                print("Did not find MMSI and IMO: " + str(mmsi_data[1]['MMSI']) + " " + str(mmsi_data[1]['IMO']))
                # still append to list so we don't search again, all values -1 as flag
                mmsi_db = mmsi_db.append({'MMSI':mmsi_data[1]['MMSI'], 'DWT':-1, 'SIZE':-1, 'DESIG':-1}, ignore_index=True)
        else:
            print("Failed to retrieve page " + str(tgt))

    # all done, write out mmsi_db
    mmsi_db.to_csv(os.path.join(cwd,mmsiDB), index=False)

if __name__ == "__main__":
    # everything should be saved to 1 db file at the end
    cwd = os.getcwd()
    # main take 2 command line args, first is the csv with the mmsis to search for, should be single column csv with only mmsis in it
    # second arg is the mmsi DB file name to store data in
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file',   type=str,   help="Name of csv file containing MMSIs and IMOs to search for")
    parser.add_argument('--output_file',  type=str,   help="Name of csv file to save results to")

    args = parser.parse_args()
    main(args.input_file, args.output_file, cwd)

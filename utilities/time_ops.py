
import datetime
import re

def file_date(fname, precision='second', ret='start_date'):
    # For new dataset
    
    if precision=='second':
        element_list = re.split('_|\.',fname)
        file_end_date = datetime.datetime.strptime(element_list[-3], "%Y%m%d-%H%M%S")
        file_start_date = datetime.datetime.strptime(element_list[-4], "%Y%m%d-%H%M%S")
    elif precision=='date':
        element_list = re.split('_|-|\.',fname)
        file_end_date = datetime.datetime.strptime(element_list[-4], "%Y%m%d")
        file_start_date = datetime.datetime.strptime(element_list[-6], "%Y%m%d")
    else:
        print("Please specify second or date as the level of precision")
    if ret=='start_date':
        return(file_start_date)
    else:
        return file_start_date, file_end_date



    
def utc_to_date(utc):
    date = datetime.datetime.utcfromtimestamp(utc)
    return date
    


'''
    # Use for old datasets (without include/discard flags)
    if precision=='second':
        element_list = re.split('_|\.',fname)
        file_end_date = datetime.datetime.strptime(element_list[-2], "%Y%m%d-%H%M%S")
        file_start_date = datetime.datetime.strptime(element_list[-3], "%Y%m%d-%H%M%S")
    elif precision=='date':
        element_list = re.split('_|-|\.',fname)
        file_end_date = datetime.datetime.strptime(element_list[-3], "%Y%m%d")
        file_start_date = datetime.datetime.strptime(element_list[-5], "%Y%m%d")
'''
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

#DeepTrap
from DeepTrap import BackgroundSubtraction, create_h5, utils

def convert_time(time_string):
    try:
        date_object = datetime.strptime(time_string , "%Y-%m-%d %H:%M:%S")
    except:
        return "day"
    
    #time interval
    is_day = date_object.hour  > 8 and date_object.hour  < 17
    
    if is_day:
        return "day"
    else:
        return "night"
    
def sort_locations(data):
    """Divide the input data into location sets for background subtraction"""
    
    #denote day and night, if poorly formatted add to day.
    data["day_night"] = data.date_captured.apply(convert_time)

    #Split into nested dict by location and daynight
    location_dict = {}
    for i in data["location"].unique():
        location_data = data[data.location == i]
        location_dict[i] = {}        
        for j in location_data["day_night"].unique():
            location_dict[i][j] = location_data[location_data.day_night == j]
    
    return location_dict

def preprocess_location(location_data, config, destination_dir):
    """A dictionary object with keys day and night split by pandas image data"""
    
    #Create h5 file for location holder
    location = location_data["day"].location.unique()[0]      
    
    #Create an h5 and csv file
    n_images = location_data["day"].shape[0] + location_data["night"].shape[0] 
    image_shape = (config["height"], config["width"])
    
    h5_file, csv_file = create_h5.create_files(
        destination_dir,
        location,
        image_shape = image_shape,
        n_images = n_images) 
    
    h5_index = 0 
    for day_or_night in location_data:
        
        #Selection image data
        image_data = location_data[day_or_night]
        
        #sort by timestamp
        image_data = image_data.sort_values("date_captured")
        
        #Create a background model object for camera location of sequences
        bgmodel = BackgroundSubtraction.BackgroundModel(image_data, target_shape = image_shape)
        
        #Select representative images based on temporal median difference
        #place outputs in a csv and h5 holder position
        if bgmodel:
            h5_index = bgmodel.run(h5_file, csv_file, h5_index)
        else:
            print("File exists, skipping location")
        
    #close files
    #report h5 file size
    nfiles = len(h5_file["images"])
    fname = h5_file.filename
    
    csv_file.close()
    h5_file.close()
    
    print("{} file exists with {} files".format(fname, nfiles))
    return "{} file exists with {} files".format(fname, nfiles)
        
        

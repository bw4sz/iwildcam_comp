import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

#DeepTrap
if __name__ == "__main__":
    import BackgroundSubtraction, create_h5, utils
else:
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

def preprocess_location(location_data, config, destination_dir, training=True):
    """A dictionary object with keys day and night split by pandas image data"""
    
    location_images = []    
    location_labels = []
    location_filenames = []
    
    for day_or_night in location_data:
        #Selection image data
        image_data = location_data[day_or_night]
        
        #sort by timestamp
        image_data = image_data.sort_values("date_captured")
        
        #Create a background model object for camera location of sequences
        bgmodel = BackgroundSubtraction.BackgroundModel(image_data, day_or_night = day_or_night, training=training, destination_dir = destination_dir, config=config)
        
        #Select representative images based on temporal median difference
        #Create an h5 holder
        if bgmodel:
            message = bgmodel.run()
        else:
            message = "File exists, skipping location"
        
    print(message)
    return message

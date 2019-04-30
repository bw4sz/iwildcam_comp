import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import BackgroundSubtraction
import utils
import Detector
import create_h5

#Read and log config file
config = utils.read_config(prepend="..")
output_dir = config["output_dir"]

debug=True
#check for image dir
#if not os.path.exists(output_dir):
    #os.mkdir(output)
    
#use local image copy
if debug:
    config["train_data_path"] = "../tests/data/sample_location"
    config["h5_dir"] = "/Users/Ben/Downloads/"

#Load data
train_df = pd.read_csv('../data/train.csv')
train_df['file_path'] = train_df['id'].apply(lambda x: os.path.join(config["train_data_path"], f'{x}.jpg'))
train_df = utils.check_images(train_df, config["train_data_path"])

#Sort images into location
locations  = BackgroundSubtraction.sort_locations(train_df)

for location in locations:
    location_images = []    
    location_labels = []
    location_filenames = []
    for day_or_night in locations[location]:
            
        #Selection image data
        image_data = locations[location][day_or_night]
        
        #sort by timestamp
        image_data = image_data.sort_values("date_captured")
        
        #Create a background model object for camera location of sequences
        bgmodel = BackgroundSubtraction.BackgroundModel(image_data, day_or_night = day_or_night )
        
        #Select representative images based on temporal median difference
        images, labels, filenames = bgmodel.run()
        
        #Add to location lists
        location_images.append(images)
        location_labels.append(labels)
        location_filenames.append(filenames)
    
    #Write h5 file
    create_h5.generate(location_images, location_labels, location_filenames, destination_dir, location)
    
    
    
    

        
#Save predicted empty based on temporal median
#Convert to dataframe and save
#predicted_empty

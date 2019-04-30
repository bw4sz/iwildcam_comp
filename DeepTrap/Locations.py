import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import BackgroundSubtraction
import utils
import Detector

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

#Load data
train_df = pd.read_csv('../data/train.csv')
train_df['file_path'] = train_df['id'].apply(lambda x: os.path.join(config["train_data_path"], f'{x}.jpg'))
train_df = utils.check_images(train_df, config["train_data_path"])

#Sort images into location
locations  = BackgroundSubtraction.sort_locations(train_df)

predicted_empty = []
predicted_boxes = []

for location in locations:
    for day_or_night in locations[location]:
            
        #Selection image data
        image_data = locations[location][day_or_night]
        
        #sort by timestamp
        image_data = image_data.sort_values("date_captured")
        
        #Create a background model object for camera location of sequences
        bgmodel = BackgroundSubtraction.BackgroundModel(image_data, day_or_night = day_or_night )
        
        #Select representative images based on temporal median difference
        target_images = bgmodel.run()
        
        for sequence in target_images:
            image_dict = target_images[sequence]
            for file_name in image_dict:
                image_crop = Detector.run_crop(file_name = file_name , box=image_dict[file_name])
                
            #crop images based on Megadetector intersecting with temporal median box
        
        #Write tfrecords
        
        #Side effect, those with no boxes are predicted empty
        predicted_empty.append(bgmodel.predictions)
        
#Save predicted empty based on temporal median
#Convert to dataframe and save
#predicted_empty

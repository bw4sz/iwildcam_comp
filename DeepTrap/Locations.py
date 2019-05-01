import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#DeepTrap
import BackgroundSubtraction
import utils
import Detector
import create_h5

def preprocess_location(location_data, destination_dir):
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
        bgmodel = BackgroundSubtraction.BackgroundModel(image_data, day_or_night = day_or_night )
        
        #Select representative images based on temporal median difference
        images, labels, filenames = bgmodel.run()
        
        #Add to location lists
        location_images.append(images)
        location_labels.append(labels)
        location_filenames.append(filenames)
    
    #Write h5 file
    #Flatten out lists
    location_images = [item for sublist in location_images for item in sublist]
    location_labels = [item for sublist in location_labels for item in sublist]
    location_filenames = [item for sublist in location_filenames for item in sublist]
    
    create_h5.generate(location_images, location_labels, location_filenames, destination_dir= destination_dir, location=location)

if __name__=="__main__":
    #Read and log config file
    config = utils.read_config(prepend="..")
    debug=True
    
    #use local image copy
    if debug:
        config["train_data_path"] = "../tests/data/sample_location"
        config["train_h5_dir"] = "/Users/Ben/Downloads/train/"
        config["test_h5_dir"] = "/Users/Ben/Downloads/test/"
        
    
    destination_dir = config["train_h5_dir"] 
    #check for image dir
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
            
    #Load train data
    train_df = pd.read_csv('../data/train.csv')
    train_df['file_path'] = train_df['id'].apply(lambda x: os.path.join(config["train_data_path"], f'{x}.jpg'))
    train_df = utils.check_images(train_df, config["train_data_path"])
    
    #Sort images into location
    locations  = BackgroundSubtraction.sort_locations(train_df)
        
    for location in locations:
        location_data = locations[location]
        preprocess_location(location_data, destination_dir)    
    
    #test data
    test_df = pd.read_csv('../data/test.csv')
    test_df['file_path'] = test_df['id'].apply(lambda x: os.path.join(config["test_data_path"], f'{x}.jpg'))
    test_df = utils.check_images(test_df, config["test_data_path"])
    
    destination_dir = config["test_h5_dir"] 
    #check for image dir
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
        
    #Sort images into location
    locations  = BackgroundSubtraction.sort_locations(test_df)
        
    for location in locations:
        location_data = locations[location]
        preprocess_location(location_data, destination_dir)
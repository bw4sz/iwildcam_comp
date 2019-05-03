import sys
import pytest
import os
import pandas as pd

#Path hack - how to make this more portable?
sys.path.append('/Users/ben/Documents/iwildcam_comp/')
from ..DeepTrap import Locations, utils

def test_preprocess_location():
    #Read and log config file
    config = utils.read_config(prepend="..")
    debug=True
    
    #use local image copy
    if debug:
        config["train_data_path"] = "../tests/data/iWildCam_2019_CCT/iWildCam_2019_CCT_images"
        config["test_data_path"] = "../tests/data/iWildCam_2019_IDFG/iWildCam_IDFG_images"
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
    locations  = Locations.sort_locations(train_df)
    
    results = []
    for location in locations:
        location_data = locations[location]
        message = Locations.preprocess_location(location_data, destination_dir=destination_dir, config=config)    
        results.append(message)
        
    #test data
    test_df = pd.read_csv('../data/test.csv')
    test_df['file_path'] = test_df['id'].apply(lambda x: os.path.join(config["test_data_path"], f'{x}.jpg'))
    test_df = utils.check_images(test_df, config["test_data_path"])
    
    destination_dir = config["test_h5_dir"] 
    #check for image dir
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
        
    #Sort images into location
    locations  = Locations.sort_locations(test_df)
        
    for location in locations:
        location_data = locations[location]
        results = Locations.preprocess_location(location_data, config, destination_dir)    
    
test_preprocess_location()
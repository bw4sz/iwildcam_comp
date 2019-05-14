import sys
import pytest
import os
import pandas as pd
import h5py

#Path hack - how to make this more portable?
sys.path.append('/Users/ben/Documents/iwildcam_comp/')
from ..DeepTrap import Locations, utils

def test_preprocess_location():
    #Read and log config file
    config = utils.read_config(prepend="..")
    debug=True
    
    #use local image copys
    if debug:
        config["train_data_path"] = "../tests/data/iWildCam_2019_CCT/iWildCam_2019_CCT_images/"
        config["test_data_path"] = "../tests/data/iWildCam_2019_IDFG/iWildCam_IDFG_images/"
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
    
    #Grab known location to test
    location_data = locations[21]
    message = Locations.preprocess_location(location_data, destination_dir=destination_dir, config=config)    
        
    #Read and test results
    sample_csv = pd.read_csv(os.path.join(destination_dir,"21.csv"))
    sample_h5 = h5py.File(os.path.join(destination_dir,"21.h5"))
    
    #test length
    assert sample_csv.shape[0] == train_df[train_df.location==21].shape[0], "csv shape does not match input data"
    assert len(sample_h5["images"]) == train_df[train_df.location==21].shape[0], "h5 shape does not match input data"
    
test_preprocess_location()
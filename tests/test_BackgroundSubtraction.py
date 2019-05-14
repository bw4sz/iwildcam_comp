#Test a long running background subtraction - memory use seems high.
import pytest
import sys
import os
import pandas as pd

#Path hack - how to make this more portable?
sys.path.append('/Users/ben/Documents/iwildcam_comp/')
from ..DeepTrap import utils, Locations

##test data
config = utils.read_config(prepend="..")
config["train_data_path"] = "data/iWildCam_2019_CCT/iWildCam_2019_CCT_images"
config["test_data_path"] = "data/iWildCam_2019_IDFG/iWildCam_IDFG_images/"
config["train_h5_dir"] = "/Users/Ben/Downloads/train/"
config["test_h5_dir"] = "/Users/Ben/Downloads/test/"    

test_df = pd.read_csv('../data/test.csv')
test_df['file_path'] = test_df['id'].apply(lambda x: os.path.join(config["test_data_path"], f'{x}.jpg'))
test_df = utils.check_images(test_df, config["test_data_path"])
destination_dir = config["test_h5_dir"] 

#check for image dir
if not os.path.exists(destination_dir):
    os.mkdir(destination_dir)
    
#Sort images into location
locations  = Locations.sort_locations(test_df)
order= test_df.groupby("location").size().sort_values(ascending=False).index.values
Locations.preprocess_location(locations[73], config, destination_dir)

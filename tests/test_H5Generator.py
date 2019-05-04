import sys
import pytest
import os
import pandas as pd
import numpy as np

#Path hack - how to make this more portable?
sys.path.append('/Users/ben/Documents/iwildcam_comp/')
from ..DeepTrap import utils, H5Generator

def test_Generator():
    config = utils.read_config(prepend="..")
    config["train_data_path"] = "data/iWildCam_2019_CCT/iWildCam_2019_CCT_images"
    config["test_data_path"] = "data/iWildCam_2019_IDFG/iWildCam_IDFG_images/"
    config["classification_model"]["epochs"] = 1
    config["classification_model"]["batch_size"] =3
    config["classification_model"]["gpus"] = 1
    config["train_h5_dir"] = "/Users/Ben/Downloads/train/"
    config["test_h5_dir"] = "/Users/Ben/Downloads/test/"    
    
    #Load train data
    train_df = pd.read_csv('../data/train.csv')
    train_df['file_path'] = train_df['id'].apply(lambda x: os.path.join(config["train_data_path"], f'{x}.jpg'))
    train_df = utils.check_images(train_df, config["train_data_path"])
 
    training_split, evaluation_split = utils.split_training(train_df, image_dir=config["train_data_path"])
 
    #Known size
    assert training_split.shape == (26, 13), "Training split wrong size"
    
    train_generator = H5Generator.Generator(training_split, 
                                batch_size=config["classification_model"]["batch_size"], 
                                h5_dir=config["train_h5_dir"])
    
    #Try a sample image load
    sample_id = "59817a83-23d2-11e8-a6a3-ec086b02610b.jpg"
    image = train_generator.load_image(sample_id)
    assert image.shape == (config["width"], config["height"], 3), "Image shape does not match config"
    
    label = train_generator.load_annotation(sample_id)
    assert train_generator.name_to_label(np.argmax(label)) == "mountain_lion", "label does not match"
    
    train_generator.plot_image(sample_id, "This should be a mountain lion")
    
    #Try a sample __getitem__
    assert len(train_generator.__getitem__(0))==2, "Batch does not yield 2 items"
        
test_Generator()
     

#Utilities for loading data
import yaml
import json
import pandas as pd
import os

classes = {0: "empty", 1:"deer", 2:"moose", 3:"squirrel", 4:"rodent",
    5:"small_mammal",6: "elk", 7:"pronghorn_antelope", 8:"rabbit",9: "bighorn_sheep", 10:"fox",11: "coyote", 
    12:"black_bear", 13:"raccoon", 14: "skunk", 15: "wolf", 16:"bobcat", 17: "cat",
    18:"dog", 19:"opossum", 20: "bison", 21: "mountain_goat", 22:"mountain_lion"}

def read_config():
    with open("config.yaml") as f:
        config = yaml.load(f)
    return config
    
def read_train_data(image_dir, supp_data=False):
    train_df = pd.read_csv('data/train.csv')
    train_df['file_path'] = train_df['id'].apply(lambda x: os.path.join(image_dir, f'{x}.jpg'))
    train_df['dataset'] = "train"    
    train_df['class'] = train_df['category_id'].apply(lambda x: classes[x])
    
    if supp_data:
        print("Loading supp data")
        supp_data_df = read_supp_data()
        
    return train_df

def read_supp_data():
    
    supp_df =  pd.read_json('data/iWildCam_2019_iNat_Idaho.json', orient="split")
    with open('data/iWildCam_2019_iNat_Idaho.json') as json_data:
        supp_data = json.load(json_data)
    
    image_df = pd.DataFrame.from_dict(supp_data['images'])
    image_df = image_df[['file_name','id']]
    image_df.rename(columns={'id':'image_id'}, inplace=True)   
    
    annotation_df = pd.DataFrame.from_dict(supp_data['annotations'])
    full_df = image_df.merge(annotation_df)
    
    #match to class names
    full_df['class'] = full_df['category_id'].apply(lambda x: classes[x])
    
    return full_df

def read_test_data(image_dir):
    test_df = pd.read_csv('data/test.csv')
    test_df['file_path'] = test_df['id'].apply(lambda x: os.path.join(image_dir,f'{x}.jpg'))
    test_df['dataset'] = "test"    
        
    return test_df

def submission(predictions):
    submission_df = pd.read_csv('data/sample_submission.csv')    
    submission_df['Predicted'] = predictions.values
    submission_df.head()    
    submission_df.to_csv("output/submission.csv",index=False)
    
    return submission_df
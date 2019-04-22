#Utilities for loading data
import yaml
import json
import pandas as pd
import os

def label_to_class():
    classes = """empty, 0
    deer, 1
    moose, 2
    squirrel, 3
    rodent, 4
    small_mammal, 5
    elk, 6
    pronghorn_antelope, 7
    rabbit, 8
    bighorn_sheep, 9
    fox, 10
    coyote, 11
    black_bear, 12
    raccoon, 13
    skunk, 14
    wolf, 15
    bobcat, 16
    cat, 17
    dog, 18
    opossum, 19
    bison, 20
    mountain_goat, 21
    mountain_lion, 22""".split('\n')
    classes = {int(i.split(', ')[1]): i.split(', ')[0] for i in classes}
    
    return classes

classes = label_to_class()

def read_config():
    with open("config.yaml") as f:
        config = yaml.load(f)
    return config
    
def read_train_data(supp_data=False):
    train_df = pd.read_csv('data/train.csv')
    train_df['file_path'] = train_df['id'].apply(lambda x: f'{x}.jpg')
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

def read_test_data():
    test_df = pd.read_csv('data/test.csv')
    test_df['file_path'] = test_df['id'].apply(lambda x: f'{x}.jpg')
    test_df['dataset'] = "test"    
        
    return test_df

def submission(predictions):
    submission_df = pd.read_csv('data/sample_submission.csv')    
    submission_df['Predicted'] = predictions.values
    submission_df.head()    
    submission_df.to_csv("output/submission.csv",index=False)
    
    return submission
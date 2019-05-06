#Utilities for loading data
import yaml
import json
import pandas as pd
import os
import glob
import h5py

classes = {0: "empty", 1:"deer", 2:"moose", 3:"squirrel", 4:"rodent",
    5:"small_mammal",6: "elk", 7:"pronghorn_antelope", 8:"rabbit",9: "bighorn_sheep", 10:"fox",11: "coyote", 
    12:"black_bear", 13:"raccoon", 14: "skunk", 15: "wolf", 16:"bobcat", 17: "cat",
    18:"dog", 19:"opossum", 20: "bison", 21: "mountain_goat", 22:"mountain_lion"}

def check_h5s(data, h5_dir):
    existing_h5s = glob.glob(os.path.join(h5_dir,"*.h5"))
    
    #Check if they can be opened
    finished = []
    for h5_filename in existing_h5s:
        try:
            hdf5_file = h5py.File(h5_filename, mode='r')    
            finished.append(h5_filename)
        except Exception as e:
            print("Error: {}, Removing location {}, can't open file".format(e, h5_filename))
            
    existing_locations = [os.path.splitext(os.path.basename(x))[0] for x in finished]
    existing_locations = [int(x) for x in existing_locations]
    data = data[data.location.isin(existing_locations)]
    
    return data
    
def check_images(data, image_dir):
    #get available images
    image_paths = glob.glob(os.path.join(image_dir,"*.jpg"))
    data = data[data.file_path.isin(image_paths)].reset_index()
    assert data.shape[0] != 0, "Data is empty, check image path {}".format(image_dir)    
    
    return data
    
def read_config(prepend=None):
    filename="config.yaml"
    if prepend:
        filename = os.path.join(prepend,filename)
    with open(filename) as f:
        config = yaml.load(f)
    return config
    
def read_train_data(image_dir):
    train_df = pd.read_csv('data/train.csv')
    train_df['file_path'] = train_df['id'].apply(lambda x: os.path.join(image_dir, f'{x}.jpg'))
    train_df['dataset'] = "train"    
    train_df['class'] = train_df['category_id'].apply(lambda x: classes[x])
    
    train_df = train_df.drop_duplicates(subset="file_name")
    
    #animal_empty ID, 0 for empty
    train_df["is_animal"] = train_df["category_id"] != 0
    
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
        
    test_df = test_df.drop_duplicates(subset="file_name")
        
    return test_df


def split_training(train_df, image_dir):
    """Split the training data based on camera locations, as this most closely mirrors the competition goal"""
    #split on location data
    percent_training = 0.75
    unique_locations = train_df.location.drop_duplicates()
    
    #Split the first rows
    training_locations = unique_locations.head(n=int(unique_locations.shape[0]*percent_training))    
    training_split = train_df[train_df.location.isin(training_locations)]
    evaluation_split  = train_df[~ train_df.location.isin(training_locations)]
    
    return training_split, evaluation_split
 
def filter_training(training_split):
    """Intelligent undersampling to reduce class imbalance"""
    empty_seq = training_split[training_split.category_id==0].groupby("location").apply(lambda x: x.seq_id.drop_duplicates().head(200))
    empty_images = training_split[training_split.seq_id.isin(empty_seq) & (training_split.category_id==0)].groupby("seq_id").apply(lambda x: x.sample(1))

    #add back in animals
    animal_images = training_split[training_split.category_id!=0]
    training_split = pd.concat([empty_images, animal_images])     
    
    return training_split
 
def sequence_voting(prediction_data):
    """prediction_data: a pandas frame with a "prediction" column containing the category id, and seq id"""
        
    #Two rules
    #0 Assumption: creating only a single label per sequence.
    #1 - For animal categories perform majority rule
    animal_predictions = prediction_data[prediction_data.predictions != 0]    
    
    #Get each sequence and find most common animal prediction
    most_common = animal_predictions.groupby("seq_id").apply(lambda x: top_class_no_ties(x))
    
    #Returning NaN causes all sorts of int -> float type changes, do -99 and remove.
    most_common = most_common[most_common!=-99].to_dict()
    
    #make a copy of the frame to update from
    new_predictions = prediction_data.copy()
    new_predictions["predictions"] = new_predictions["seq_id"].map(most_common)
    new_predictions = new_predictions.dropna()
    
    #drop NaN and reset type
    prediction_data.update(new_predictions)
    
    #keeps setting to float for some reason
    prediction_data.predictions = prediction_data.predictions.astype("int")
    
    return prediction_data

def top_class_no_ties(data):
    """Helper function to pick the top class of animal, but leave alone if there are ties"""

    #count each category, sorts by descending
    dcounts = data.predictions.value_counts()
    
    #If there is more than one category, is there a tie? A bit ugly code
    if len(dcounts.values) > 1:
        if dcounts.values[0] == dcounts.values[1]:
            return -99
        else:
            top_class  = dcounts.head(1).index.values[0]
    else:
        top_class  = dcounts.head(1).index.values[0]
        
    return top_class
    
def submission(predictions, datestamp):
    submission_df = pd.read_csv('data/sample_submission.csv')    
    submission_df['Predicted'] = predictions
    submission_df.head()    
    submission_df.to_csv("output/{}_submission.csv".format(datestamp),index=False)
    
    return submission_df
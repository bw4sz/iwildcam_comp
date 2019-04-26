#Download sample location
import yaml
import pandas as pd
import numpy as np
import os
from vassal.terminal import Terminal

#Load training data
train_df = pd.read_csv('../data/train.csv')

#select a sequence for each class
file_paths = train_df[train_df.location == 21].file_name

def read_config(prepend=None):
    filename="config.yaml"
    if prepend:
        filename = os.path.join(prepend,filename)
    with open(filename) as f:
        config = yaml.load(f)
    return config

#set data path
config = read_config(prepend ="..")

fullpath = [os.path.join(config["train_data_path"],x) for x in file_paths]
fullpath_df = pd.DataFrame(fullpath)

#loop through and download files using scp
for file_path in fullpath:
    cwd = os.getcwd()
    cmd ="scp b.weinstein@hpg2.rc.ufl.edu:{} {}/data/sample_location/".format(file_path,cwd)
    shell = Terminal([cmd])
    shell.run()
    
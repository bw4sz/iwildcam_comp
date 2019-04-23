from DeepTrap import utils
import pandas as pd
import numpy as np
import os
from vassal.terminal import Terminal

#Load training data
train_df = utils.read_train_data()

#select a sequence for each class
sequences = train_df.groupby("category_id").apply(lambda x: x.sample(1)).seq_id
file_paths = train_df[train_df.seq_id.isin(sequences)].file_path

#set data path
config = utils.read_config()

fullpath = [os.path.join(config["train_data_path"],x) for x in file_paths]
fullpath_df = pd.DataFrame(fullpath)
fullpath_df.to_csv("tests/files_to_download.csv")

#loop through and download files using scp
for file_path in fullpath:
    cwd = os.getcwd()
    cmd ="scp b.weinstein@hpg2.rc.ufl.edu:{} {}/tests/test_data/iWildCam_2019_CCT/iWildCam_2019_CCT_images/".format(file_path,cwd)
    shell = Terminal([cmd])
    shell.run()
    
#Test data - no labels, so just download 5 random sequences
#Load training data
test_df = utils.read_test_data()

#select a sequence for each class
sequences = test_df.seq_id.sample(5)
file_paths = test_df[test_df.seq_id.isin(sequences)].file_path
fullpath = [os.path.join(config["test_data_path"],x) for x in file_paths]
fullpath_df = pd.DataFrame(fullpath)

#loop through and download files using scp
for file_path in fullpath:
    cwd = os.getcwd()
    cmd ="scp b.weinstein@hpg2.rc.ufl.edu:{} {}/tests/test_data/iWildCam_2019_IDFG/iWildCam_IDFG_images/".format(file_path,cwd)
    shell = Terminal([cmd])
    shell.run()

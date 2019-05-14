import comet_ml
experiment = comet_ml.Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='iwildcam_comp', log_code=True)

#set matplotlib
import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
from datetime import datetime
import pandas as pd

from DeepTrap import utils
from DeepTrap.models import resnet
from DeepTrap import evaluation, visualization, callback
from DeepTrap.H5Generator import Generator

#Set training or training
mode_parser = argparse.ArgumentParser(description='DeepTrap Training')
mode_parser.add_argument('--debug', action="store_true")
mode_parser.add_argument("-g", "--gpus", type=int, default=1, help="# of GPUs to use for training")
mode =mode_parser.parse_args()

#Output dir
dirname = datetime.now().strftime("%Y%m%d_%H%M%S")

#Read and log config file
config = utils.read_config()

#log
experiment.log_parameters(config)
experiment.log_parameters(prefix= "classification_model", dic=config["classification_model"])
experiment.log_parameters(prefix= "bgmodel", dic=config["bgmodel"])

#use local image copy
if mode.debug:
    config["train_data_path"] = "tests/data/iWildCam_2019_CCT/iWildCam_2019_CCT_images"
    config["test_data_path"] = "tests/data/iWildCam_2019_IDFG/iWildCam_IDFG_images/"
    config["classification_model"]["epochs"] = 1
    config["classification_model"]["batch_size"] =3
    config["classification_model"]["gpus"] = 1
    config["train_h5_dir"] = "/Users/Ben/Downloads/train/"
    config["test_h5_dir"] = "/Users/Ben/Downloads/test/"    
    
#load annotations
train_df = utils.read_train_data(image_dir=config["train_data_path"])

#Ensure images exist
train_df = utils.check_images(train_df, config["train_data_path"])

#check h5 preprocessed utils.check_h5
train_df = utils.check_h5s(train_df, config["train_h5_dir"])

#Create keras training generator - split the training data into a validation set, both from the California site.
training_split, evaluation_split = utils.split_training(train_df, image_dir=config["train_data_path"])

#remove empty from set for testing.
#Try to minimize sources of risk here, just take a set of images from both
if not mode.debug:
    #Rule 1, minimize empty frames, but keep as much generalization as possible. take a random sample of each empty sequence, and then random sample of locations
    training_split = utils.filter_training(training_split)
    
    #evaluation_split = evaluation_split[evaluation_split.category_id.isin([0,1,10])].groupby("category_id",as_index=False).apply(lambda x: x.head(10))

experiment.log_parameter("Training Images", training_split.shape[0])

#Log 
train_generator = Generator(training_split, 
                            batch_size=config["classification_model"]["batch_size"], 
                            h5_dir=config["train_h5_dir"],
                            image_dir=config["train_data_path"])

#Sanity check
assert train_generator.size() > 0, "No training data available"

evaluation_generator = Generator(evaluation_split,
                            batch_size=config["classification_model"]["batch_size"], 
                            h5_dir=config["train_h5_dir"], image_dir=config["train_data_path"])

experiment.log_parameter("Evaluation Images", evaluation_generator.size())
assert evaluation_generator.size() > 0, "No evaluation data available"

#Create callbacks
evalution_callback = callback.Evaluate(evaluation_generator, experiment)

#Load Model
model = resnet.Model(config)

#Train Model
model.train(train_generator, evaluation_generator=evaluation_generator, callbacks=[evalution_callback])

#Predict evaluation data
#Test data
test_df = utils.read_test_data(image_dir=config["test_data_path"])
test_df = utils.check_images(test_df, config["test_data_path"])
test_df = utils.check_h5s(test_df, config["test_h5_dir"])

if not mode.debug:
    test_df = test_df.sample(n=100)

#Create evaluation generator and predict
validation_generator = Generator(test_df,
                                 batch_size=config["classification_model"]["batch_size"], 
                                 h5_dir=config["test_h5_dir"],
                                 training=False,
                                 image_dir=config["test_data_path"])
#predict
predictions = model.predict(validation_generator)

#View predictions
#turn to classes from one-hot label
predictions = np.argmax(predictions, axis=1)
predictions_label = [utils.classes[x] for x in predictions]

if mode.debug:
    visualization.plot_images(validation_generator, predictions=predictions_label, n=2,annotations=False, show=False, experiment=experiment)
else:
    visualization.plot_images(validation_generator, predictions=predictions_label,n=100, annotations=False, show=False, experiment=experiment)    
 
#submission doc
#if not mode.debug:
    #submission_df = utils.submission(predictions, dirname)
    #experiment.log_asset(file_like_object=submission_df)    

experiment.end()


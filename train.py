import comet_ml
experiment = comet_ml.Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='iwildcam_comp', log_code=True)

#set matplotlib
import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
from datetime import datetime

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
experiment.log_parameters(config)
experiment.log_parameters(prefix= "classification_model", dic=config["classification_model"])
experiment.log_parameters(prefix= "bgmodel", dic=config["bgmodel"])

#use local image copy
if mode.debug:
    config["train_data_path"] = "tests/data/sample_location/"
    config["test_data_path"] = "tests/data/iWildCam_2019_IDFG/iWildCam_IDFG_images/"
    config["classification_model"]["epochs"] = 1
    config["classification_model"]["batch_size"] =3
    config["classification_model"]["gpu"] = 1
    config["h5_dir"] = "/Users/Ben/Downloads/"
    
#load annotations
train_df = utils.read_train_data(image_dir=config["train_data_path"], supp_data=False)

#Ensure images exist
train_df = utils.check_images(train_df, config["train_data_path"])

#Create keras training generator - split the training data into a validation set, both from the California site.
training_split, evaluation_split = utils.split_training(train_df, image_dir=config["train_data_path"] )

#reduce training locations for temporary model checking, stop at 10000 images, but keep full locations.
location_filter = training_split.groupby("location").size().sort_values().cumsum() < 50000 
selected_locations = location_filter[location_filter==True].index.values
training_split = training_split[training_split.location.isin(selected_locations)]

#remove empty from set for testing.
#training_split = training_split[training_split.category_id!=0]
#evaluation_split =evaluation_split[evaluation_split.category_id!=0]

#Log m
train_generator = Generator(training_split, 
                            batch_size=config["classification_model"]["batch_size"], 
                            h5_dir=config["h5_dir"])

evaluation_generator = Generator(evaluation_split,
                            batch_size=config["classification_model"]["batch_size"], 
                            h5_dir=config["h5_dir"])

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

if not mode.debug:
    test_df = test_df.sample(n=100)

#Create evaluation generator and predict
validation_generator = Generator(test_df,
                                 image_size=config["classification_model"]["image_size"],
                                 batch_size=config["classification_model"]["batch_size"], 
                                 image_dir=config["test_data_path"],
                                 training=False)
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


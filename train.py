import comet_ml
experiment = comet_ml.Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='iwildcam_comp', log_code=True)
import argparse
import numpy as np
from datetime import datetime

from DeepTrap import utils
from DeepTrap.models import resnet
from DeepTrap import evaluation, visualization, callback
from DeepTrap.Generator import Generator

#Set training or training
mode_parser = argparse.ArgumentParser(description='DeepTrap Trainin')
mode_parser.add_argument('--debug', action="store_true")
mode_parser.add_argument("-g", "--gpus", type=int, default=1, help="# of GPUs to use for training")
mode =mode_parser.parse_args()

#Output dir
dirname = datetime.now().strftime("%Y%m%d_%H%M%S")

#Read and log config file
config = utils.read_config()
experiment.log_parameters(config)

#use local image copy
if mode.debug:
    config["train_data_path"] = "tests/data/iWildCam_2019_CCT/iWildCam_2019_CCT_images/"
    config["test_data_path"] = "tests/data/iWildCam_2019_IDFG/iWildCam_IDFG_images/"
    config["epochs"] = 1
    config["batch_size"] =1
    config["gpu"] = 1
    
#load annotations
train_df = utils.read_train_data(image_dir=config["train_data_path"], supp_data=False)
test_df = utils.read_test_data(image_dir=config["test_data_path"])

#Create keras training generator - split the training data into a validation set, both from the California site.
training_split, evaluation_split = utils.split_training(train_df, image_dir=config["train_data_path"] )
train_generator = Generator(training_split, config=config, image_dir=config["train_data_path"])
evaluation_generator = Generator(evaluation_split, config=config, image_dir=config["train_data_path"])

#if debug:
    #visualization.plot_images(train_generator, n=5,annotations=True, show=True)

#Create callbacks
evalution_callback = callback.Evaluate(evaluation_generator, experiment)

#Load Model
model = resnet.Model(config)

#Train Model
model.train(train_generator, evaluation_generator=evaluation_generator, callbacks=[evalution_callback])

#Predict evaluation data
#Create evaluation generator for Idaho Data
validation_generator = Generator(test_df, config=config, image_dir=config["test_data_path"],training=False)
predictions = model.predict(validation_generator)

#View predictions
#turn to classes from one-hot label
predictions = np.argmax(predictions, axis=1)
predictions_label = [utils.classes[x] for x in predictions]

if mode.debug:
    visualization.plot_images(validation_generator, predictions=predictions_label, n=2,annotations=False, show=False, experiment=experiment)
else:
    visualization.plot_images(validation_generator, predictions=predictions_label,n=50, annotations=False, show=False, experiment=experiment)    
 
#submission doc
if not mode.debug:
    submission_df = utils.submission(predictions, dirname)
    experiment.log_asset(file_like_object=submission_df)

experiment.end()


#import logging dashboard
#import comet_ml
#experiment = comet_ml.Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='iwildcam_comp', log_code=True)
from DeepTrap import utils
from DeepTrap.models import random
from DeepTrap import evaluation, visualization
from DeepTrap.Generator import Generator

#TODO set in argparse.
debug = True

#Read config file
config = utils.read_config()

#use local image copy
if debug:
    config["train_data_path"] = "tests/data/iWildCam_2019_CCT/iWildCam_2019_CCT_images/"
    config["test_data_path"] = "tests/data/iWildCam_2019_IDFG/iWildCam_IDFG_images/"
    
#load annotations
train_df = utils.read_train_data(image_dir=config["train_data_path"], supp_data=False)
test_df = utils.read_test_data(image_dir=config["test_data_path"])

#Create keras training generator
train_generator = Generator(train_df, config=config, image_dir=config["train_data_path"])
visualization.plot_images(train_generator, n= 5,annotations=True, show=True)

#Create callbacks
#callbacks = callback.create(train_generator,config)

#Load Model
model = random.Model(config)

#Train Model
model.train(train_generator)

#Predict evaluation data
#Create evaluation generator for Idaho Data
validation_generator = Generator(test_df, config=config, image_dir=config["test_data_path"])
predictions = model.predict(validation_generator)

#View predictions
#turn to classes
predictions_label = [utils.classes[x] for x in predictions]
visualization.plot_images(validation_generator, predictions=predictions_label, n= 2,annotations=False, show=True)

#submission doc
if not debug:
    submission_df = utils.submission(predictions)
#log
#experiment.log_asset("output/submission.csv")


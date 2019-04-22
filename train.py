#import logging dashboard
#import comet_ml
#experiment = comet_ml.Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2", project_name='iwildcam_comp', log_code=True)
from DeepTrap import utils
from DeepTrap.models import random
from DeepTrap import evaluation

#Read config file
config = utils.read_config()

#load annotations
train_df = utils.read_train_data(supp_data=False)
test_df = utils.read_test_data()

#Create keras generator
generator = Generator(train_df, config)

#Create callbacks
#callbacks = callback.create(generator,config)

#Load Model
model = random.Model(config)

#Train Model
model.train(train_df)

#Predict evaluation data
predictions = model.predict(test_df)

#submission doc
submission_df = utils.submission(predictions)
#log
#experiment.log_asset("output/submission.csv")


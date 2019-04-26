#MobileNet for fast background subtraction empty v 

import numpy as np
import keras
from keras.preprocessing import image

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

class Model():
    
    def __init__(self, config):
        self.model = keras.applications.mobilenet.MobileNet()
    
    def train(self, train_generator, callbacks=None, evaluation_generator=None):
                
        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_generator.size()/self.config["bgmodel"]["batch_size"],
            epochs=self.config["bgmodel"]["epochs"],
            verbose=2,
            validation_data=evaluation_generator,
            shuffle=False,
            callbacks=callbacks,
        use_multiprocessing = False)       
            
    def predict(self, generator):
        predictions = self.model.predict_generator(generator, steps=None, max_queue_size=10, verbose=1)
        return predictions
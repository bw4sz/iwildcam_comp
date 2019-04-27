#Deep Residual Network
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.models import Model as Keras_Model
from keras.layers import Dense, GlobalAveragePooling2D

import keras
from keras.utils import get_file
import tensorflow as tf
from .. import utils, preprocess

class Model():
    
    def __init__(self, config):
        self.config = config
        self.image_size = config["classification_model"]["image_size"]
        shape = (self.image_size , self.image_size , 3)
        self.num_classes = len(utils.classes)
        
        #Define input shape
        self.input_shape = keras.layers.Input(shape)
        
        #if multiple gpu
        num_gpu = config["classification_model"]["gpus"] 
        if num_gpu > 1:
            from keras.utils import multi_gpu_model
            with tf.device('/cpu:0'):     
                self.model = self.load_model()                                
            self.model = multi_gpu_model(self.model, gpus=num_gpu)
        else:
            self.model = self.load_model()            
        
        #compile
        self.model.compile("adam", "categorical_crossentropy", ["accuracy"])

    def load_model(self):
        
        #load pretrained model
        base_model = ResNet50(input_tensor=self.input_shape, weights='imagenet', include_top=False)        
        
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        #fully-connected layer
        x = Dense(1024, activation='relu')(x)
        
        #Softmax prediction
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Keras_Model(inputs=base_model.input, outputs=predictions)
        
        return model
        
    def train(self, train_generator, callbacks=None, evaluation_generator=None):
                
        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_generator.size()/self.config["classification_model"]["batch_size"],
            epochs=self.config["classification_model"]["epochs"],
            verbose=2,
            validation_data=evaluation_generator,
            shuffle=False,
            callbacks=callbacks,
        use_multiprocessing = False)       
            
    def predict(self, generator):
        
        #Check the remainder of the batch size, do batches that fit
        predictions_batches = self.model.predict_generator(generator, max_queue_size=10, 
                                    workers=1, 
                                    use_multiprocessing=False, 
                                    verbose=1)
        
        #construction the final batch seperately
        predictions = predictions_batches[:generator.size(),:]
        
        return predictions
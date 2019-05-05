#Deep Residual Network
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.models import Model as Keras_Model
from keras.layers import Dense, GlobalAveragePooling2D

import keras
from keras.utils import get_file
import tensorflow as tf
from .. import utils, preprocess
from keras import backend as K

from sklearn.utils import class_weight

class Model():
    
    def __init__(self, config):
        self.config = config
        self.image_height = config["height"]
        self.image_width = config["width"]
        shape = (self.image_height , self.image_width , 3)
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
        
        #compile focal loss
        self.model.compile("adam", [categorical_focal_loss(alpha=.25, gamma=2)], ["accuracy"])
        
        #compile
        #self.model.compile("adam", "categorical_crossentropy", ["accuracy"])

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
            verbose=1,
            validation_data=evaluation_generator,
            shuffle=False,
            callbacks=callbacks,
            use_multiprocessing = False,
            workers=2,
            max_queue_size=10)       
            
    def predict(self, generator):
        
        #Check the remainder of the batch size, do batches that fit
        predictions_batches = self.model.predict_generator(generator, max_queue_size=20, 
                                    workers=2, 
                                    use_multiprocessing=True, 
                                    verbose=1)
        
        #construction the final batch seperately
        predictions = predictions_batches[:generator.size(),:]
        
        return predictions
    
#Define focal loss 
def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed
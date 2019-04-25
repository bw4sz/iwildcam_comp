#Deep Residual Network
import keras_resnet.models
import keras
from keras.utils import get_file

from .. import utils


class Model():
    
    def __init__(self, config):
        self.config = config
        self.image_size = config["image_size"]
        shape, classes = (self.image_size , self.image_size , 3), len(utils.classes)
        
        #Define input shape
        x = keras.layers.Input(shape)
        
        #if multiple gpu
        num_gpu = config["gpus"] 
        if num_gpu > 1:
            from keras.utils import multi_gpu_model
            with tf.device('/cpu:0'):
                self.model = keras_resnet.models.ResNet50(x, classes=classes)                
                self.model = multi_gpu_model(model, gpus=num_gpu)
        else:
            self.model = keras_resnet.models.ResNet50(x, classes=classes)
            self.model.compile("adam", "categorical_crossentropy", ["accuracy"])
        
        #Load imagenet weights
        imagenet_weights = self.download_imagenet()
        self.model.load_weights(imagenet_weights, by_name=True, skip_mismatch=True)
    
    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        From https://github.com/fizyr/keras-retinanet/blob/9f6ee78a67d27e5d6d9055c2a0da3803e03b290b/keras_retinanet/models/resnet.py
        """
        resnet_filename = 'ResNet-{}-model.keras.h5'
        resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
        #depth = int(self.backbone.replace('resnet', '')) #TODO allow different depths?
        depth=50

        filename = resnet_filename.format(depth)
        resource = resnet_resource.format(depth)
        if depth == 50:
            checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
        elif depth == 101:
            checksum = '05dc86924389e5b401a9ea0348a3213c'
        elif depth == 152:
            checksum = '6ee11ef2b135592f8031058820bb9e71'

        return get_file(
            filename,
            resource,
            cache_subdir='models',
            md5_hash=checksum
        )
    
    def train(self, train_generator, callbacks=None, evaluation_generator=None):
                
        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_generator.size()/self.config["batch_size"],
            epochs=self.config["epochs"],
            verbose=2,
            validation_data=evaluation_generator,
            shuffle=False,
            callbacks=callbacks,
        )        
            
    def predict(self, generator):
        predictions = self.model.predict_generator(generator)
        return predictions
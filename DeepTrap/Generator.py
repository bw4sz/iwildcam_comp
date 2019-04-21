"""
Generator to stream images into model
"""
import keras
import DeepTrap 
from keras_retinanet.preprocessing.generator import Generator as retinanet_generator

class Generator(retinanet_generator):
    """ Generate data for a custom dataset.
    """

    def __init__(
        self,
        data,
        annotations,
        config,
        group_method="none",
        **kwargs
    ):
        """ Initialize a data self.
        """
        
        #Assign config and intiliaze values
        self.config
        self.data = data
        self.annotations = annotations
        
        #Read classes
        self.classes=self.read_classes()  
        
        #Create label dict
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key                
        
        super(Generator, self).__init__(**kwargs)
                        
    def __len__(self):
        """Number of batches for self"""
        return len(self.groups)
         
    def size(self):
        """ Size of the dataset.
        """
        return self.data.shape()[0]
    
    def read_classes(self):
        """ 
        Number of annotation classes
        """
        
        number_of_classes = len(self.annotations.label.unique())

        return(number_of_classes)
    
    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]
        
    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        #Select sliding window and tile
        image_name = self.image_names[image_index]        
        self.row = self.image_data[image_name]
        
        #
        self.read_image()
        
        return image
    
    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        #Find the original data and crop
        image_name = self.image_names[image_index]
        self.row = self.image_data[image_name]
        
        #load boxes
        

        return boxes
    

        

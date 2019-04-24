"""
Generator to stream images into model
"""
import os
import numpy as np
import keras
import random
from DeepTrap.utils import classes
from DeepTrap import preprocess
from DeepTrap.visualization import draw_annotation
from PIL import Image
import glob

class Generator(keras.utils.Sequence):
    """ Generate data for a custom dataset.
    Inspired by fizyr/retinanet https://github.com/fizyr/keras-retinanet/
    """

    def __init__(
        self,
        train_df,
        image_dir,
        config,
        training=True
    ):
        """ Initialize a data self.
        """
        
        #Assign config and intiliaze values
        self.config = config
        self.data = train_df
        self.image_dir = image_dir
        self.training=training
        
        #Read classes
        self.classes=classes
        
        #Create label dict
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key                
        
        #Check for images - limit training data to those images which are present
        self.check_images()
        
        #Create indices
        self.define_groups()
        
    def __len__(self):
        """Number of batches for self"""
        return len(self.groups)
    
    def check_images(self):
        #get available images
        image_paths = glob.glob(os.path.join(self.image_dir,"*.jpg"))
        self.data = self.data[self.data.file_path.isin(image_paths)].reset_index()
        assert self.data.shape[0] != 0, "Data is empty, check image path {}".format(self.image_dir)

    def define_groups(self, shuffle = True):
        self.image_dict = self.data.to_dict("index")
        order = list(self.image_dict.keys())
        #Shuffle input order
        if shuffle:
            random.shuffle(order)
        
        #Split into batches
        self.groups = [[order[x % len(order)] for x in range(i, i + self.config["batch_size"])] for i in range(0, len(order), self.config["batch_size"])]
    
    def size(self):
        """ Size of the dataset.
        """
        return self.data.shape[0]
    
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
        #Load an image from file
        filename = self.image_dict[image_index]["file_path"]                
        im = Image.open(filename)
        self.image = np.array(im)
        return self.image
    
    def load_annotation(self, image_index):
        """ Load annotations for an image_index.
        """
        #Find the original data and crop
        self.label = self.image_dict[image_index]["category_id"]  
        
        #turn to categorical? not sure
        categorical_label = keras.utils.np_utils.to_categorical(self.label, num_classes=len(self.classes))
        
        return categorical_label
    
    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        annotations_group = [self.load_annotation(image_index) for image_index in group]
        annotations_group = np.stack(annotations_group)
        
        return annotations_group
    
    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]
     
    def preprocess_group(self, image_group):
        """ Preprocess image and its annotations.
        """
        for i in range(len(image_group)):
            # preprocess the image
            image = preprocess.preprocess_image(image_group[i])
    
            # resize image
            image_group[i] = preprocess.resize_image(image, size=self.config["image_size"])
        
        return image_group
        
    def plot_image(self, image_index, label):
        """plot current image"""
        self.load_image(image_index)
        fig = draw_annotation(self.image, label)
        return fig
    
    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.config["batch_size"],) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image
            
        return image_batch
            
    def __getitem__(self, index):
        """Yield the next batch of images"""
        #Select group
        group = self.groups[index]
        
        #Load images
        image_group = self.load_image_group(group)
        
        #Normalize and resize
        image_group = self.preprocess_group(image_group)
        
        #Create a batch object
        image_batch = self.compute_inputs(image_group)
        
        #If training generator, load annotations
        if self.training:
            annotations = self.load_annotations_group(group)
            return image_batch, annotations
        else:
            return image_batch
        
        
        

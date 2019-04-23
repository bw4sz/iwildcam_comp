"""
Generator to stream images into model
"""
import os
import numpy as np
import keras
import random
from DeepTrap.utils import classes
from DeepTrap.visualization import draw_annotation
from PIL import Image
import glob

class Generator(keras.utils.Sequence):
    """ Generate data for a custom dataset.
    """

    def __init__(
        self,
        train_df,
        image_dir,
        config,
    ):
        """ Initialize a data self.
        """
        
        #Assign config and intiliaze values
        self.config = config
        self.data = train_df
        self.image_dir = image_dir
        
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
        return self.label
    
    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        annotations_group = [self.load_annotation(image_index) for image_index in group]
        return annotations_group
    
    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]
     
    def plot_image(self, image_index, label):
        self.load_image(image_index)
        fig = draw_annotation(self.image, label)
         
    def __getitem__(self):
        """Yield the next batch of images"""
        group = self.groups[index]
        self.load_group(group)
        
        
        

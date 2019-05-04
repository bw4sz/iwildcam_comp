"""
Generator to stream images into model
"""
import os
import numpy as np
import keras
import random
from PIL import Image
import glob
import h5py
from matplotlib import pyplot as plt 
import cv2

from DeepTrap.utils import classes as classification_classes
from DeepTrap import preprocess

class Generator(keras.utils.Sequence):
    """ Generate data for a custom dataset.
    Inspired by fizyr/retinanet https://github.com/fizyr/keras-retinanet/
    data: a pandas dataframe from utils.read
    image_dir: path to directory of image files
    batch_size: training batch size
    image size: resize image to square
    classes: a dict of class labels key -> name
    training: whether annotations should be loaded.
    """

    def __init__(
        self,
        data,
        h5_dir,
        image_dir,
        batch_size,
        classes=classification_classes,
        training=True
    ):
        """ Initialize a data self.
        """
        
        #Assign config and intiliaze values
        self.data = data
        self.h5_dir = h5_dir
        self.image_dir = image_dir
        self.training=training
        self.batch_size = batch_size

        #Read classes
        self.classes = classes
        
        #Create label dict
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key                
    
        #Create indices
        self.define_groups()
        
        #Init has an empty location
        self.previous_location = None
        
    def __len__(self):
        """Number of batches for self"""
        return len(self.groups)

    def define_groups(self, shuffle = True):
        
        #Sort by location
        self.data = self.data.sort_values("location").reset_index()
        self.image_dict = self.data.set_index("file_name").to_dict("index")
        order = list(self.image_dict.keys())
        
        #Shuffle input order
        if shuffle:
            random.shuffle(order)
        
        #Split into batches
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
    
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
        #Load an image from file based on location
        location = self.image_dict[image_index]["location"]
        filename = self.image_dict[image_index]["file_name"]                        
        
        if location != self.previous_location:
            h5_name = os.path.join(self.h5_dir,"{}.h5".format(location))
            self.hf = h5py.File(h5_name, 'r')
            
            #load filename csv
            csv_name = os.path.join(self.h5_dir,"{}.csv".format(location))
            self.filename_csv = pd.DataFrame.from_csv(csv_name)
            
            #reset location for easy loading
            self.previous_location = location            
        
        #Reading file_name from h5, it needs to be decoded
        h5_index = self.filename_csv[filename==self.filename_csv.filename].h5_index.values[0]
        
        #Load image
        self.image = self.hf["images"][h5_index,...]
        
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
            image_group[i] = preprocess.preprocess_image(image_group[i])
    
            # resize image
            #image_group[i] = preprocess.resize_image(image, size=self.image_size)
        
        return image_group
        
    def plot_image(self, image_index, title):
        """plot current image"""
        
        #Show background subtracted image
        fig=plt.figure()
        image = self.load_image(image_index)
        plt.subplot(1,2,1)
        plt.imshow(image)
        
        #Show original
        original_path_name = os.path.join(self.image_dir,image_index)
        original = cv2.imread(original_path_name)
        rgb_original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        plt.subplot(1,2,2)
        plt.imshow(rgb_original)
        plt.title(title)        
        return fig
    
    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

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
        
        
        

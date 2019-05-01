#H5 file for precomputing background subtracted data
import os
import h5py
import numpy as np
import cv2

from DeepTrap import utils
config = utils.read_config()
    
def resize(image, image_shape):
    height, width, channels = image_shape
    image = cv2.resize(image, (width, height))
    return image
    
def generate(images, labels, filenames, destination_dir, location):
    """Create and save a precomputed location of images
    images,labels,filenames are lists of results from Location.py
    destination dir: output dir to save h5 file
    location: a name for the filename """
    
    #Assume that all files have the same shape
    image_shape = (config["height"], config["width"], 3)
    hdf5_file = create_file(destination_dir, location, image_shape, n_images=len(images))
    if not hdf5_file:
        return "{} already exists".format(location)
    write_records(hdf5_file, images, labels, filenames, image_shape)
    
    #Print message
    h5_filename = os.path.join(destination_dir, str(location) + ".h5")
    
    return "{} created".format(h5_filename)
    
def create_file(destination_dir, location, image_shape, n_images, overwrite=False):
    """Create a h5 in the directory specified named for the location. h5 has two objects, the images and the labels"""
    #Create h5 dataset    
    # open a hdf5 file and create arrays
    h5_filename = os.path.join(destination_dir, str(location) + ".h5")
    
    if not overwrite:
        file_exists = os.path.exists(h5_filename)
        return None
        
    hdf5_file = h5py.File(h5_filename, mode='w')    
        
    #Create h5 dataset to fill
    height, width, channels = image_shape
    hdf5_file.create_dataset("images", (n_images, height, width, channels), dtype='int')
    hdf5_file.create_dataset("labels", (n_images,1) , dtype='int')
    hdf5_file.create_dataset("filenames", (n_images, 1), dtype='S10')
    
    return hdf5_file

def write_records(hdf5_file, images,labels,filenames, image_shape):
    """lists of the images, labels and filenames to hold in an h5 container"""
    
    for i in range(len(images)):
        
        #if not the desired size, resize
        
        if images[i].shape != image_shape:
            images[i] = resize(images[i], image_shape)
        hdf5_file["images"][i,...] = images[i]
        hdf5_file["labels"][i,...] = labels[i]
        hdf5_file["filenames"][i,...] = np.string_(filenames[i])
    
    hdf5_file.close()

    
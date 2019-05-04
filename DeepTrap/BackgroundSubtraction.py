#Collect images by location
import pandas as pd
import cv2
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

#DeepTrap
from DeepTrap import create_h5

#helper function
def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
    d2 = datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
    return abs((d2 - d1).days)

#Start a background subtraction object
       
class BackgroundModel():
    
    def __init__(self,image_data, day_or_night, destination_dir, config):
        """Create a background model class
        image_data: pandas dataframe of image data
        day_or_night: is the sequence at "night" or 'day' to help set settings
        return a saved h5 file to disk with outputs for training/prediction
        """
        self.data = image_data
        
        #Set a global sequence background
        self.sequence_background = None
        
        #White balancing
        self.wb = cv2.xphoto.createSimpleWB()
        self.wb.setP(0.4)
        
        #Create h5 container for results
        location = image_data.location.unique()[0]  
        n_images = image_data.shape[0]
        self.image_shape = (config["height"], config["width"])
        
        self.h5_file = create_h5.create_file(
            destination_dir,
            location,
            image_shape = self.image_shape,
            n_images = n_images) 
        
        #If overwrite and file exists, exit.
        if self.h5_file is None:
            return None
        
    def split_sequences(self):
        """Divide pandas dataframe into dictionary of sequences of images"""
        #unique ids
        unique_ids = list(self.data.seq_id.unique())
        sequence_dict = {}
        
        #split data
        for id in unique_ids:
            sequence_dict[id] = self.data[self.data.seq_id == id]
        return sequence_dict
            
    def plot_sequence(self, image_data):
        #Get file paths
        files= list(image_data.file_path)
        rows = 2
        self.fig = plt.figure()
        
        #Loop and plot
        for index, path in enumerate(files):
            plt.subplot(2,len(files),index+1)
            image = cv2.imread(path)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)            
            plt.imshow(img_rgb)
            
    def load_image(self, path):
        """Load and image and convert color space"""
        img = cv2.imread(path)
        img = self.preprocess(img)
        
        #Check size        
        return img
        
    def preprocess(self, img):
        """White balance and change color space"""
        img = self.wb.balanceWhite(img)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)  
                
        return img
    
    def resize_sequence(self, images, shape):
        """for a set of images, resize to target size if needed
        images: a list of numpy images
        shape: a numpy shape target object
        """
        for index, image in enumerate(images):
            if image.shape != shape:
                height, width, channels = shape
                images[index] = cv2.resize(image, (width, height))
        return images
        
    def create_background(self, image_data, target_shape):  
        """
        Generate a temporal median image
        image_data: pandas dataframe
        target_shape: reshape if within location shape varies among images.
        """
        images = []
        for index, row in image_data.iterrows():
            img = self.load_image(row.file_path)
            images.append(img)
        
        #Check image shapes, optionally resize
        images = self.resize_sequence(images, target_shape)        
        
        #Stack images into a single array
        if len(images) == 0:
            return None
        
        images = np.stack(images)
        median_background = np.median(images, axis=0)
        median_background = median_background.astype(np.uint8)
        return median_background
    
    def post_process(self,image):
        """Assumes YCrCB color space
        """
        #Scale just the luminance, clamp negative values
        #image[:,:,0] = cv2.subtract(image[:,:,0], image[:,:,0].mean())
        
        image = cv2.blur(image, (3,3))
    
        #divide by max and scale to 0-255 for each channel
        image[:,:,0] = image[:,:,0] / image[:,:,0].max() * 255
        image[:,:,1] = image[:,:,1] / image[:,:,1].max() * 255
        image[:,:,2] = image[:,:,2] / image[:,:,2].max() * 255
        
        return image
    
    def apply(self, background, image):
        """subtract an image from background and convert to colorspace"""
        #Background subtractions
        foreground = cv2.absdiff(background, image)
        
        #Mean center
        thresh = self.post_process(foreground)
        
        return thresh
        
    def run_sequence(self, image_data):
        """apply background subtraction to a set of images
        return a doct the image (with boxes) in the sequence with the largest bounding box {file_name} -> box
        """
        #list file paths - skip first image
        images_to_run = list(image_data.file_path)
        num_images = len(images_to_run)
        
        #Container for output boxes
        subtracted_images = [ ]
        filenames = []
        
        for index, image_path in enumerate(images_to_run):
            #Load image
            image = self.load_image(image_path)
            
            #temporal median 
            sequence_background = self.create_background(image_data[image_data.file_path !=image_path], target_shape=image.shape)
            
            #image threshold - threshold based on day night
            threshold_image = self.apply(sequence_background, image)
            subtracted_images.append(threshold_image)
            
            #grab filename
            filename = image_data[image_data.file_path == image_path].file_name.values[0]
            filenames.append(filename)
            
            #plot
            #plt.subplot(2,num_images,num_images + index+1)
            #back_to_rgb = cv2.cvtColor(threshold_image, cv2.COLOR_BGR2RGB)
            #plt.imshow(back_to_rgb)
        #plt.show()                
                           
        return (subtracted_images, filenames)
        
    def run_single(self, image_data):
        """image_data: The sequence level pandas data table"""
        
        #list file paths - skip first image
        images_to_run = list(image_data.file_path)
        num_images = len(images_to_run)
        
        for index, image_path in enumerate(images_to_run):
            #Loag target image
            image = self.load_image(image_path)
            
            #Create a background from the entire location image_object, grabbing the 6 closest frames in time?
            background_data = self.data[self.data.file_path != image_path]
            if background_data.shape[0] < 5:
                    threshold_image = image
            else:
                target_date = self.data[self.data.file_path == image_path].date_captured.values[0]
                background_data["date_diff"] = background_data.date_captured.apply(lambda x: days_between(x,target_date))
                background_data=background_data.sort_values("date_diff").head(5) 
                
                #Remove taget image
                self.sequence_background = self.create_background(background_data, target_shape=image.shape)
                                   
                #image threshold
                threshold_image = self.apply(self.sequence_background, image)
                
            #grab filename
            filename = image_data[image_data.file_path == image_path].file_name.values[0]            
                
        ##plot
        #plt.subplot(2,num_images,num_images + index+1)            
        #back_to_rgb = cv2.cvtColor(threshold_image, cv2.COLOR_BGR2RGB)
        #plt.imshow(back_to_rgb)
        #plt.show()
        
        return ([threshold_image], [filename])
        
    
    def write_h5(self, images, filenames):
        """write a list of images and filenames from a sequence"""
        create_h5.write_records(self.h5_file, images, filenames, 
                               self.image_shape)
        
    def run(self):
        
        #split into sequences
        sequence_dict = self.split_sequences()
        print("{} sequences found".format(len(sequence_dict)))
        
        #target images container
        for sequence in sequence_dict:
            
            #Get image data
            image_data = sequence_dict[sequence]
           
            #Burst set of images?
            is_sequence = image_data.shape[0] > 1

            #self.plot_sequence(image_data)                  
            if is_sequence:
                #Run background subtraction
                seq_images, seq_filenames = self.run_sequence(image_data)
            else:
                #Get a global background model and individual image
                seq_images, seq_filenames = self.run_single(image_data)
                
            #write to h5
            self.write_h5(seq_images, seq_filenames)
            
        #report h5 file size
        nfiles = len(self.h5_file["filenames"])
        fname = self.h5_file.filename
        self.h5_file.close()
        return "{} file exists with {} files".format(fname, nfiles)
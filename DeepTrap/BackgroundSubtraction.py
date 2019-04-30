#Collect images by location
import pandas as pd
import cv2
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

#Start a background subtraction object
def sort_locations(data):
    """Divide the input data into location sets for background subtraction"""
    
    #denote day and night, if poorly formatted add to day.
    for index, row in data.iterrows():
        try:
            date_object = datetime.strptime(data.date_captured[index] , "%Y-%m-%d %H:%M:%S")
        except:
            data.at[index,'day_night'] = "day"
        is_day = date_object.hour  > 8 and date_object.hour  < 17
        if is_day:
            data.at[index,'day_night'] = "day"
        else:
            data.at[index,'day_night'] = "night"

        #Split into nested dict by location and daynight
    location_dict = {}
    for i in data["location"].unique():
        location_data = data[data.location == i]
        location_dict[i] = {}        
        for j in location_data["day_night"].unique():
            location_dict[i][j] = location_data[location_data.day_night == j]
    
    return location_dict
       
class BackgroundModel():
    
    def __init__(self,image_data, day_or_night):
        """Create a background model class
        image_data: pandas dataframe of image data
        day_or_night: is the sequence at "night" or 'day' to help set settings
        """
        
        self.data = image_data
        
        #White balancing
        self.wb = cv2.xphoto.createSimpleWB()
        self.wb.setP(0.4)
        
        #Empty predictions
        self.predictions = {}
        
        #Box predictions
        self.box_predictions = {}
        
        #Set min threshold
        if day_or_night == "day":
            self.min_threshold = 30
        else:
            self.min_threshold = 5
        
    def split_sequences(self):
        """Divide pandas dataframe into dictionary of sequences of images"""
        #unique ids
        unique_ids = list(self.data.seq_id.unique())
        sequence_dict = {}
        
        #split data
        for id in unique_ids:
            sequence_dict[id] = self.data[self.data.seq_id == id]
        return sequence_dict
    
    def load_image(self, path):
        """Load and image and convert color space"""
        img = cv2.imread(path)
        img = self.preprocess(img)
        
        #Check size        
        return img
        
    def preprocess(self, img):
        """White balance and change color space"""
        img_wb = self.wb.balanceWhite(img)
        img_YCrCb = cv2.cvtColor(img_wb,cv2.COLOR_BGR2YCrCb)  
                
        return img_YCrCb
    
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
        images = np.stack(images)
        median_background = np.median(images, axis=0)
        median_background = median_background.astype(np.uint8)
        return median_background
    
    def post_process(self,image):
        """Assumes YCrCB color space
        """
        
        #Scale just the luminance
        image[:,:,0] = image[:,:,0]  - image[:,:,0].mean()
    
        #remove negative values
        image *= (image>0)
        
        #divide by max and scale to 0-255 for each channel
        image[:,:,0] = image[:,:,0] / image[:,:,0].max() * 255
        image[:,:,1] = image[:,:,1] / image[:,:,1].max() * 255
        image[:,:,2] = image[:,:,2] / image[:,:,2].max() * 255
        
        image = cv2.medianBlur(image, 7)
        
        return image
    
    def apply(self, background, image, min_threshold):
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
        labels = []
        filenames = []
        
        for index, image_path in enumerate(images_to_run):
            #Load image
            image = self.load_image(image_path)
            
            #temporal median 
            sequence_background = self.create_background(image_data[image_data.file_path !=image_path], target_shape=image.shape)
            
            #image threshold - threshold based on day night
            threshold_image = self.apply(sequence_background, image, self.min_threshold )
            subtracted_images.append(threshold_image)
            
            #grab label
            label = image_data[image_data.file_path == image_path].category_id.values[0]
            filename = image_data[image_data.file_path == image_path].file_name.values[0]
            
            labels.append(label)
            filenames.append(filename)
            
            #plot
            #plt.subplot(2,num_images,num_images + index+1)                
            #plt.imshow(threshold_image[:,:,0:])
        #plt.show()                
                           
        return (subtracted_images, labels, filenames)
        
    def run_single(self, image_data):
        """image_data: The sequence level pandas data table"""
        
        #list file paths - skip first image
        images_to_run = list(image_data.file_path)
        num_images = len(images_to_run)
        
        for index, image_path in enumerate(images_to_run):
            #Loag target image
            image = self.load_image(image_path)
            
            #Create a background from the entire location image_object, except for target image
            background_data = self.data[self.data.file_path != image_path]
            
            #There can be alot of images in location, limit for sake of memory
            if background_data.shape[0] > 100:
                background_data = background_data.sample(n=100)
                
            #Remove taget image
            sequence_background = self.create_background(background_data, target_shape=image.shape)
            
            #image threshold
            threshold_image = self.apply(sequence_background, image, min_threshold=self.min_threshold)
            
            #grab label and filename
            label = image_data[image_data.file_path == image_path].category_id.values[0]
            filename = image_data[image_data.file_path == image_path].file_name.values[0]            
            
        return ([threshold_image], [label], [filename])
                
        ##plot
        #plt.subplot(2,num_images,num_images + index+1)                
        #plt.imshow(threshold_image[:,:,0:])
        #plt.show()
        
    def run(self):
        
        #Container for target images and labels
        images = []
        labels = []
        filenames = []
        
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
                seq_images, seq_labels, seq_filenames = self.run_sequence(image_data)
            else:
                #Get a global background model and individual image
                seq_images, seq_labels, seq_filenames = self.run_single(image_data)
                
            #Add to a flat list of results
            for i in range(len(seq_images)):
                images.append(seq_images[i])
                labels.append(seq_labels[i])
                filenames.append(seq_filenames[i])

        return images, labels, filenames
    
    def draw_box(self, image, boxes):
        for bounding_box in boxes:
            cv2.rectangle(image, (bounding_box.x, bounding_box.y+bounding_box.h),
                          (bounding_box.x + bounding_box.w, bounding_box.y), (255,0,0), 5)
        return image
            
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
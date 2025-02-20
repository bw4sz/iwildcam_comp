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
    try:
        d1 = datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
        d2 = datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
        day_diff = abs((d2 - d1).days)
    except:
        print("error in date transform, setting to large value (100)")
        day_diff = 0
    return day_diff

#Start a background subtraction object
class BackgroundModel():
    
    def __init__(self,image_data, target_shape, plot=False):
        """Create a background model class
        image_data: pandas dataframe of image data
        return A Object of Class Background Model
        """
        self.data = image_data
        self.target_shape = target_shape
        self.plot = plot
        
        #Set a global sequence background
        self.sequence_background = None
        
        #White balancing
        self.wb = cv2.xphoto.createSimpleWB()
        self.wb.setP(0.4)
        
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
    
    def post_process(self,difference_image, original_image):
        """Assumes YCrCB color space
        """
            
        #divide by max and scale to 0-255 for each channel
        difference_image[:,:,0] = difference_image[:,:,0] / difference_image[:,:,0].max() * 255
        difference_image[:,:,1] = difference_image[:,:,1] / difference_image[:,:,1].max() * 255
        difference_image[:,:,2] = difference_image[:,:,2] / difference_image[:,:,2].max() * 255
        
        #Threshold        
        #Mask and get the original colors
        im_gray = cv2.cvtColor(difference_image, cv2.COLOR_BGR2GRAY)        
        _, mask = cv2.threshold(im_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)    
        
        #Remove noise in mask, open then close holes.
        kernel = np.ones((5,5),np.uint8)        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((9,9),np.uint8)                
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        #Get 3 color images
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        im_thresh_color = cv2.bitwise_and(original_image, mask3)
        
        return im_thresh_color
    
    def apply(self, background, image):
        """subtract an image from background and convert to colorspace"""
        #Background subtractions
        foreground = cv2.absdiff(background, image)
        
        #Mean center and threshold
        thresh = self.post_process(foreground, image)
        
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
            
            #if sequence is longer than 6 images, sample it
            background_data = image_data[image_data.file_path !=image_path]
            
            if background_data.shape[0] > 6:
                background_data = background_data.sample(6)
                       
            #temporal median 
            sequence_background = self.create_background(background_data, target_shape=image.shape)
            
            #image threshold - threshold based on day night
            threshold_image = self.apply(sequence_background, image)
            subtracted_images.append(threshold_image)
            
            #grab filename
            filename = image_data[image_data.file_path == image_path].file_name.values[0]
            filenames.append(filename)
            
            #plot
            if self.plot:
                plt.subplot(2,num_images,num_images + index+1)
                back_to_rgb = cv2.cvtColor(threshold_image, cv2.COLOR_BGR2RGB)
                plt.imshow(back_to_rgb)
        if self.plot:
            plt.show()                
                           
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
        if self.plot:
            plt.subplot(2,num_images,num_images + index+1)            
            back_to_rgb = cv2.cvtColor(threshold_image, cv2.COLOR_BGR2RGB)
            plt.imshow(back_to_rgb)
            plt.show()
        
        return ([threshold_image], [filename])
        
    def write_h5(self, h5_file, csv_file, h5_index, images, filenames, target_shape):
        """write a list of images and filenames from a sequence
        h5_index is a counter to point towards the position in the object. Should not be used in parallel, is not thread safe"""
        h5_index = create_h5.write_records(h5_file, csv_file, h5_index, images, filenames, target_shape)
        return h5_index
        
    def run(self, h5_file, csv_file, h5_index):
        """Run a location and place results in an h5 image container and csv metadata"""
        
        #split into sequences
        sequence_dict = self.split_sequences()
        print("{} sequences found".format(len(sequence_dict)))
        
        #target images container
        for sequence in sequence_dict:
                        
            #Get image data
            image_data = sequence_dict[sequence]
           
            #Burst set of images?
            is_sequence = image_data.shape[0] > 1

            if self.plot:
                self.plot_sequence(image_data)                  
            if is_sequence:
                #Run background subtraction
                seq_images, seq_filenames = self.run_sequence(image_data)
            else:
                #Get a global background model and individual image
                seq_images, seq_filenames = self.run_single(image_data)
                
            #write to h5, preserve index
            h5_index = self.write_h5(h5_file, csv_file, h5_index, seq_images, seq_filenames, self.target_shape)
        
        return h5_index
            
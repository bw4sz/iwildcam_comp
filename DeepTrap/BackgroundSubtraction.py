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
        is_day = date_object.hour  > 7 and date_object.hour  < 18
        if is_day:
            data.at[index,'day_night'] = "day"
        else:
            data.at[index,'day_night'] = "night"

        #Split into nested dict by location and daynight
    location_dict = {}
    for i in data["location"].unique():
        print(i)
        location_data = data[data.location == i]
        location_dict[i] = {}        
        for j in location_data["day_night"].unique():
            location_dict[i][j] = location_data[location_data.day_night == j]
    
    return location_dict
       
class BackgroundModel():
    
    def __init__(self,image_data):
        self.data = image_data
        self.wb = cv2.xphoto.createSimpleWB()
        self.wb.setP(0.4)        
    
    def split_sequences(self):
        #unique ids
        unique_ids = list(self.data.seq_id.unique())
        sequence_dict = {}
        
        #split data
        for id in unique_ids:
            sequence_dict[id] = self.data[self.data.seq_id == id]
        return sequence_dict
    
    def load_image(self, path):
        img = cv2.imread(path)
        img = self.preprocess(img)
        
        return img
        
    def preprocess(self, img):
        img_wb = self.wb.balanceWhite(img)
        img_gray = cv2.cvtColor(img_wb,cv2.COLOR_BGR2GRAY)  
        
        return img_gray
    
    def create_background(self, image_data):
        
        #Init model with running average and learn slowly from there.
        # setting to 32-bit floating point 
        sample_image = self.load_image(image_data.file_path.iloc[0])        
        averageValue = np.float32(sample_image)         
        
        for index, row in image_data.iterrows():
            img = self.load_image(row.file_path)
            averageValue = cv2.accumulateWeighted(img, averageValue, 0.01)
            median_background = cv2.convertScaleAbs(averageValue) 
            
        return median_background
    
    def apply(self, background, image):
        foreground = cv2.absdiff(background, image)
        thresh = cv2.threshold(foreground, 40, 255, cv2.THRESH_BINARY)[1]        
        thresh = self.post_process(thresh)
        return thresh
    
    def post_process(self,image):
        #Erode to remove noise, dilate the areas to merge bounded objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        image= cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=2)
        
        return image
    
    def run_sequence(self, image_data):
        
        #Does the image come from a sequence?
        is_sequence = image_data.shape[0] > 1
        
        #Create background
        if is_sequence:
            sequence_background = self.create_background(image_data)
            
            #list file paths - skip first image
            images_to_run = list(image_data.file_path)
        
            num_images = len(images_to_run)
            
            for index, image_path in enumerate(images_to_run):
                image = self.load_image(image_path)
                threshold_image = self.apply(sequence_background, image)
                plt.subplot(2,num_images,num_images + index+1)                
                plt.imshow(threshold_image)
            plt.show()
        else:
            print("its not a sequence")
    
    def run(self):
        
        #split into sequences
        sequence_dict = self.split_sequences()
        
        for sequence in sequence_dict:
            sequence_data = sequence_dict[sequence]
            
            #plot
            self.plot_sequence(sequence_data)
            self.run_sequence(sequence_data)
        
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
        

#Collect images by location
import pandas as pd
import cv2
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from Geometry import *

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
            self.min_threshold = 45
        else:
            self.min_threshold = 20
        
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
        
        return img
        
    def preprocess(self, img):
        """White balance and change color space"""
        img_wb = self.wb.balanceWhite(img)
        img_gray = cv2.cvtColor(img_wb,cv2.COLOR_BGR2GRAY)  
        
        return img_gray
    
    def resize_sequence(self, images, shape):
        """for a set of images, resize to target size if needed
        images: a list of numpy images
        shape: a numpy shape target object
        """
        for index, image in enumerate(images):
            if image.shape != shape:
                height, width = shape
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
        images = np.dstack(images)
        median_background = np.median(images, axis=2)
        median_background = median_background.astype(np.uint8)
        return median_background
    
    def post_process(self,image):
        #Erode to remove noise, dilate the areas to merge bounded objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
        image= cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=2)
        
        return image
    
    def get_parent_bounding_box(self, bounding_boxes, index):
        for bounding_box in bounding_boxes:
            if index in bounding_box.members:
                return bounding_box
            
        return None
    
    def find_contours(self, image):
        """Compute contours from threshold images and remove small boxes"""
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
        contours = [contour for contour in contours if cv2.contourArea(contour) > 50]
        
        return contours
        
    def extend_rectangle(self, rect1, rect2):
        x = min(rect1.l_top.x, rect2.l_top.x)
        y = min(rect1.l_top.y, rect2.l_top.y)
        w = max(rect1.r_top.x, rect2.r_top.x) - x
        h = max(rect1.r_bot.y, rect2.r_bot.y) - y
        
        return Rect(x, y, w, h)
    
    def cluster_bounding_boxes(self, contours):
        bounding_boxes = []
        
        #For each contour, get a bounding box
        for i in range(len(contours)):
            x1,y1,w1,h1 = cv2.boundingRect(contours[i])

            parent_bounding_box = self.get_parent_bounding_box(bounding_boxes, i)
            if parent_bounding_box is None:
                parent_bounding_box = BoundingBox(Rect(x1, y1, w1, h1))
                parent_bounding_box.members.append(i)
                bounding_boxes.append(parent_bounding_box)

            for j in range(i+1, len(contours)):
                if self.get_parent_bounding_box(bounding_boxes, j) is None:
                    x2,y2,w2,h2 = cv2.boundingRect(contours[j])
                    rect = Rect(x2, y2, w2, h2)
                    distance = parent_bounding_box.rect.distance_to_rect(rect)
                    
                    #Combine close boxes.
                    if distance < 200:
                        parent_bounding_box.update_rect(self.extend_rectangle(parent_bounding_box.rect, rect))
                        parent_bounding_box.members.append(j)
                        
        return bounding_boxes
    
    def apply(self, background, image, min_threshold):
        """subtract an image from background and convert to 0-1"""
        #Background subtractions
        foreground = cv2.absdiff(background, image)
        
        #Set to 0 1. 
        thresh = cv2.threshold(foreground, min_threshold, 255, cv2.THRESH_BINARY)[1]        
        thresh = self.post_process(thresh)
        
        return thresh
    
    def find_bounding_box(self,image):
        """Draw bounding boxes for a threshold image"""
        
        contours = self.find_contours(image)
        bounding_boxes = self.cluster_bounding_boxes(contours)       
        
        return bounding_boxes
        
    def run_sequence(self, image_data):
        """apply background subtraction to a set of images"""

        #list file paths - skip first image
        images_to_run = list(image_data.file_path)
        num_images = len(images_to_run)
        
        sequence_boxes = []
        for index, image_path in enumerate(images_to_run):
            
            #Load image
            image = self.load_image(image_path)
            
            #temporal median 
            sequence_background = self.create_background(image_data[image_data.file_path !=image_path], target_shape=image.shape)
            
            #image threshold - threshold based on day night
            threshold_image = self.apply(sequence_background, image, self.min_threshold )
            
            #get bounding box and append to sequence list
            boxes = self.find_bounding_box(threshold_image)
            sequence_boxes.append(boxes)
            print("{} boxes found".format(len(boxes)))
            
            if len(boxes) > 0:
                threshold_image = self.draw_box(threshold_image, boxes)
                fname = image_data.iloc[index].file_name
                self.box_predictions[fname] = boxes
        
            #plot
            plt.subplot(2,num_images,num_images + index+1)                
            plt.imshow(threshold_image)
        plt.show()
            
        #If sequence has no boxes, predict all empty.
        if len(sequence_boxes) ==0:
            #assign 0 to entire sequence
            for fname in image_data.file_name:
                print("{} has no detection boxes, labeling sequence empty".format(fname))
                self.predictions[fname] = "0"
            

    
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
            
            #get bounding box
            boxes = self.find_bounding_box(threshold_image)
            
            print("{} boxes found".format(len(boxes)))
            
            if len(boxes) > 0:
                threshold_image = self.draw_box(threshold_image, boxes)
                fname = image_data.iloc[index].file_name
                self.box_predictions[fname] = boxes
            else:
                #assign 
                for fname in image_data.file_name:
                    print("{} has no detection boxes, labeling empty".format(fname))
                    self.predictions[fname] = "0"
                
            #plot
            plt.subplot(2,num_images,num_images + index+1)                
            plt.imshow(threshold_image)
        plt.show()
        
    def run(self):
        
        #split into sequences
        sequence_dict = self.split_sequences()
        
        for sequence in sequence_dict:
            
            #Get image data
            image_data = sequence_dict[sequence]
           
            #Burst set of images?
            is_sequence = image_data.shape[0] > 1

            self.plot_sequence(image_data)                  
            if is_sequence:
                self.run_sequence(image_data)
            else:
                #Get a global background model
                self.run_single(image_data)
                
                                          
        return self.predictions
    
    def draw_box(self, image, boxes):
        
        for bounding_box in boxes:
            cv2.rectangle(image, (bounding_box.x, bounding_box.y+bounding_box.h),
                          (bounding_box.x + bounding_box.w, bounding_box.y), 100, 4)
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

class BoundingBox:
    def update_rect(self, rect):
        self.rect = rect
        self.x = rect.l_top.x
        self.y = rect.l_top.y
        self.w = rect.width
        self.h = rect.height
        self.time=None
        self.label=(None,None)

    def __init__(self, rect):
        self.update_rect(rect)
        self.members = []
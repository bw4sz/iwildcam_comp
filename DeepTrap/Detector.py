#Megadetector
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import os
import json
import pickle
import utils
import cv2
import numpy as np
import os

config= utils.read_config(prepend="..")

DEFAULT_CONFIDENCE_THRESHOLD = 0.05

def preproccess(image):
    #TODO convert color?
    return image
    
def load_image(file_name):
    img = cv2.imread(file_name)
    #img = preproccess(file_name)
    return img
    
def read_data():
    infile = open(config["train_detections"],'rb')
    #ugly, but its the only encoding that seems to be working python3 for this object?
    train_detections = pickle.load(infile, encoding='latin1')
    infile.close()
    return train_detections

def load_image_detections(file_name, train_detections):
    """Load image detections and scores"""
    #Extract filename 
    image_id = os.path.splitext(os.path.basename(file_name))[0]
    index = np.argmax(np.array(image_id) == train_detections['images'])
    image_detection = train_detections['detections'][index]
    return image_detection

def load_image_scores(file_name, train_detections):
    image_id = os.path.splitext(os.path.basename(file_name))[0]    
    index = np.argmax(np.array(image_id) == train_detections['images'])
    image_scores = train_detections['detection_scores'][index]
    return image_scores 
    
def intersect_box():
    pass

def crop(image,box):
    #TODO crop
    return image

def plot_detection(file_name, train_detections):
    image = load_image(file_name)
    image_detection = load_image_detections(file_name, train_detections)
    image_scores = load_image_scores(file_name, train_detections)
    render_bounding_box(box=image_detection, score=image_scores, inputFileName=file_name)
    plt.show()
    
def run_crop(file_name, box):
    """
    Filename of image to crop
    Bounding box from temporal median filtering. Class Bounding Box
    """
    #Load detections
    train_detections = read_data()
    image = load_image(file_name)
    image_detection = load_image_detections(file_name, train_detections)
    image_scores = load_image_scores(file_name, train_detections)
    render_bounding_box(box=image_detection, score=image_scores, inputFileName=file_name)
    
    #detection_box = intersect_box(box, image_detection)
    
    #crop image
    #crop(image, box)
    

def render_bounding_box(box, score, inputFileName,
                        confidenceThreshold=DEFAULT_CONFIDENCE_THRESHOLD,linewidth=5):
    """
    Convenience wrapper to apply render_bounding_boxes to a single image
    """
    scores = [score]
    boxes = [box]
    render_bounding_boxes(boxes,scores,[inputFileName],
                          confidenceThreshold,linewidth)

def render_bounding_boxes(boxes, scores, inputFileNames,
                          confidenceThreshold=DEFAULT_CONFIDENCE_THRESHOLD,linewidth=5):
    """
    Render bounding boxes on the image files specified in [inputFileNames].  

    [boxes] and [scores] should be in the format returned by generate_detections, 
    specifically [top, left, bottom, right] in normalized units, where the
    origin is the upper-left.    
    """

    nImages = len(inputFileNames)
    iImage = 0

    for iImage in range(0,nImages):

        inputFileName = inputFileNames[iImage]
        image = mpimg.imread(inputFileName)
        iBox = 0; box = boxes[iImage][iBox]
        dpi = 100
        s = image.shape; imageHeight = s[0]; imageWidth = s[1]
        figsize = imageWidth / float(dpi), imageHeight / float(dpi)

        plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1])

        # Display the image
        ax.imshow(image)
        ax.set_axis_off()

        for iBox,box in enumerate(boxes[iImage]):

            score = scores[iImage][iBox]
            if score < confidenceThreshold:
                continue

            # top, left, bottom, right 
            #
            # x,y origin is the upper-left
            topRel = box[0]
            leftRel = box[1]
            bottomRel = box[2]
            rightRel = box[3]

            x = leftRel
            y = topRel
            w = (rightRel-leftRel) 
            h = (bottomRel-topRel)

            # Location is the bottom-left of the rect
            #
            # Origin is the upper-left
            iLeft = x
            iBottom = y
            rect = patches.Rectangle((y,x),w,h,linewidth=linewidth,edgecolor='r',
                                     facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)        

        # This is magic goop that removes whitespace around image plots (sort of)        
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, 
                            wspace = 0)
        plt.margins(0,0)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.axis('tight')
        ax.set(xlim=[0,imageWidth],ylim=[imageHeight,0],aspect=1)
        plt.axis('off')   
        plt.show()
        
        # plt.savefig(outputFileName, bbox_inches='tight', pad_inches=0.0, dpi=dpi, transparent=True)
        #plt.close()
        
if __name__ =="__main__":
    
    #load data
    detections = read_data()
    
    image_dir = "../tests/data/iWildCam_2019_CCT/iWildCam_2019_CCT_images/"
    #grab available files
    train_df = pd.read_csv('../data/train.csv')
    train_df['file_path'] = train_df['id'].apply(lambda x: os.path.join(image_dir, f'{x}.jpg'))            
    train_df = utils.check_images(train_df, image_dir)
    
    train_df = train_df.sort_values("date_captured")
    for index,row in train_df.iterrows():
        plot_detection(row.file_path, detections)
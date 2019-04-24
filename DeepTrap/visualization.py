#Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from DeepTrap import utils

def plot_images(generator, n=None, predictions = None, annotations = True, show=False, save=False,savepath=None, experiment=None):
    """Plot the first n images in a generator
    n: number of images; if none, all images
    annotations: Add ground truth (if available)
    predictions: a dict with predictions for each image index
    show: plot image object
    save: save image to file
    savepath: optional dir to add
    """
    if n is None:
        n = generator.size()
    
    #For each image load label
    for i in range(n):
        if annotations:
            annotation = generator.load_annotation(i)
            #Labels are in 1 hot categorical, just get index
            annotation=np.argmax(annotation)            
            annotation_class = utils.classes[annotation]
            label = "Annotation:{}".format(annotation_class)
            
        if predictions is not None:
            prediction = predictions[i]
            label="Prediction: {}".format(prediction)
        
        #Plot images
        fig = generator.plot_image(i, label)
        if show:
            plt.show()
        if save:
            if savepath is None:
                savepath = os.getcwd()
            
            #Use filename to save
            image_save_path ="{}/{}.png".format(savepath,generator.data[i]["file_name"])
            plt.savefig(image_save_path)
            
        #log on comet
        if experiment:
            experiment.log_figure(figure = fig)

#Draw annotations and labels
def draw_annotation(image, label, box=None):
    
    # Create figure and axes
    fig,ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
    
    if box:
        #Bottom left corner is xmin, ymin
        xmin, ymin, xmax, ymax =  box 
        h = ymin - ymax
        w = xmax - xmin    
        
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),h,w,linewidth=1,edgecolor='r',facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect)
        
        #add text on top left
        plt.text(xmin, ymax,label)
    else:
        plt.title(label)
    
    return fig
    
    
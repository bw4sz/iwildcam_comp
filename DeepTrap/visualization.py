#Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from DeepTrap import utils
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

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
        image_names = list(generator.image_dict.keys())
        key = image_names[i]
        if annotations:
            annotation = generator.load_annotation(key)
            
            #Labels are in 1 hot categorical, just get index
            annotation=np.argmax(annotation)            
            annotation_class = utils.classes[annotation]
            label = "Annotation:{}".format(annotation_class)
            
        if predictions is not None:
            prediction = predictions[i]
            label="Prediction: {}".format(prediction)
        
        #Plot images
        fig = generator.plot_image(key, label)
        if show:
            plt.show()
            
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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [classes[i] for i in unique_labels(y_true, y_pred)]
    if normalize:
        
        cm = cm.astype('float') / np.nansum(cm,axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, fontsize=9, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig
    
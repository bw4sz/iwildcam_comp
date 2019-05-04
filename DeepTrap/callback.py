#Callbacks
import keras
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from DeepTrap import visualization, preprocess
import matplotlib.pyplot as plt

class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """
    def __init__(self,generator, experiment=None):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.
        # Arguments
        """
        self.experiment = experiment
        self.generator = generator
        
        super(Evaluate, self).__init__()
            
    def on_epoch_end(self, epoch, logs=None):
        
        # run evaluation
        #predictions
        
        ground_truth = {}
        predictions = {}
        
        for i in self.generator.data.file_name.values:
            #Load label and predict image
            ground_truth[i] = np.argmax(self.generator.load_annotation(i))
            image = self.generator.load_image(i)
            image = preprocess.preprocess_image(image)
            
            #make into a batch of 1
            batch = np.expand_dims(image, axis=0)
            prediction= self.model.predict_on_batch(batch)
            predictions[i] = np.argmax(prediction)
        
        #Calculate f1
        results = pd.DataFrame({"filename":list(ground_truth.keys())})
        results["ground_truth"] = results.filename.map(ground_truth)
        results["predictions"] = results.filename.map(predictions)
    
        f1 = f1_score(results.ground_truth.values, results.predictions.values, average="macro")
        
        #Calculate confusion matrix
        labels = list(self.generator.classes.values())
        fig = visualization.plot_confusion_matrix(results.ground_truth, results.predictions, labels)
        if self.experiment:
            self.experiment.log_metric("f1 score", f1)        
            self.experiment.log_figure("confusion_matrix",fig)
            
        #plot 10 sample images (or max)
        images_to_plot = list(self.generator.data.groupby("category_id", as_index=False).apply(lambda x: x.sample(1)).file_name.values)
        for x in images_to_plot:
            #Find annotation
            class_label = np.argmax(self.generator.load_annotation(x))
            ground_class = self.generator.name_to_label(class_label)
            prediction_class  = self.generator.name_to_label(predictions[x])
            title = "Label: {}, Prediction {}".format(ground_class,prediction_class)
            fig = self.generator.plot_image(x, title)
            self.experiment.log_figure(x, fig,overwrite=True)
            
            #Close figure
            plt.clf()
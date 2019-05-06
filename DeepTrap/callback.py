#Callbacks
import keras
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from DeepTrap import visualization, preprocess, utils
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
        
        #TODO save predictions to file
        #self.experiment.log_other(results.to_csv("predictions_before_smoothing.csv"))
        
        #TODO smooth sequence labels based on majority rule
        prediction_data=results.merge(self.generator.data[["file_name","seq_id"]], left_on ="filename", right_on ="file_name")
        smoothed_results = utils.sequence_voting(prediction_data)
    
        f1 = f1_score(smoothed_results.ground_truth.values, smoothed_results.predictions.values, average="macro")
        
        #Calculate confusion matrix
        labels = list(self.generator.classes.values())
        fig = visualization.plot_confusion_matrix(smoothed_results.ground_truth, smoothed_results.predictions, labels)
        if self.experiment:
            self.experiment.log_metric("f1 score", f1)        
            self.experiment.log_figure("confusion_matrix",fig)
            
        #plot 10 sample images (or max)
        images_to_plot = list(self.generator.data.groupby("category_id", as_index=False).apply(lambda x: x.head(5)).file_name.values)
        for x in images_to_plot:
            #Find annotation
            class_label = np.argmax(self.generator.load_annotation(x))
            ground_class = self.generator.name_to_label(class_label)
            
            #Initial and time smoothed prediction
            initial_prediction_class  = self.generator.name_to_label(predictions[x])
            smoothed_prediction_class =  self.generator.name_to_label(prediction_data[prediction_data.file_name ==x].predictions.values[0])
            
            #Add a title plot
            title = "Label {}, Initial Prediction: {}, Smoothed Prediction {}".format(ground_class, initial_prediction_class, smoothed_prediction_class)
            fig = self.generator.plot_image(x, title)
            self.experiment.log_figure(x, fig,overwrite=True)
            
            #Close figure
            plt.clf()
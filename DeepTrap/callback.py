#Callbacks
import keras
from sklearn.metrics import f1_score
import numpy as np
from DeepTrap import visualization

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
        predictions_batches = self.model.predict_generator(self.generator)
        predictions = predictions_batches[:self.generator.size(),:]
        
        ground_truth = []
        for i in range(self.generator.size()):
            ground_truth.append(self.generator.load_annotation(i))
        
        #Calculate f1
        ground_truth = np.stack(ground_truth)
        ground_truth = np.argmax(ground_truth, axis=1)
        predictions =np.argmax(predictions, axis=1)
        
        #Sanity check
        assert ground_truth.shape == predictions.shape, "Ground truth and predictions don't have the same shape!"
        
        f1 = f1_score(ground_truth, predictions, average="macro")
        
        #Calculate confusion matrix
        labels = list(self.generator.classes.values())
        fig = visualization.plot_confusion_matrix(ground_truth, predictions, labels)
        if self.experiment:
            self.experiment.log_metric("f1 score", f1)        
            self.experiment.log_figure("confusion_matrix",fig)
            
        #plot 10 sample images (or max)
        samples_to_draw = min([50,self.generator.size()])
        images_to_plot = list(self.generator.data.groupby("category_id", as_index=False).apply(lambda x: x.head(10)).file_name.values)
        for x in images_to_plot:
            #Find index
            index = self.generator.data[self.generator.data.file_name == x].index.values[0]
            figname = self.generator.data.file_name.values[index]
            ground_class  = self.generator.name_to_label(ground_truth[index])
            prediction_class  = self.generator.name_to_label(predictions[index])
            title = "Label: {}, Prediction {}".format(ground_class,prediction_class)
            fig = self.generator.plot_image(index, title)
            self.experiment.log_figure(figname, fig,overwrite=True)
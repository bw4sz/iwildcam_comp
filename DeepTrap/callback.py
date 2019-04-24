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
        predictions = self.model.predict_generator(self.generator)
        
        ground_truth = []
        for i in range(self.generator.size()):
            ground_truth.append(self.generator.load_annotation(i))
        
        ground_truth=np.stack(ground_truth)
        
        f1 = f1_score(ground_truth, predictions,average="macro")
        
        fig = visualization.plot_confusion(ground_truth,predictions)
        
        if self.experiment:
            self.experiment.log_metric(f1)        
            self.experiment.log_figure("confusion_matrix",fig)
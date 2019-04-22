#Callbacks
import keras

class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(self):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments

        """
        super(Evaluate, self).__init__()
            
    def on_epoch_end(self, epoch, logs=None):
        
        # run evaluation
        macro_fscore = evaluate(generator)    
        
def create(generator, config):
    Evaluate(generator)
    
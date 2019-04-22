#Random Model to test baseline performance
import numpy as np
class Model():
    
    def __init__(self, config):
        self.config = config
    
    def train(self, training_data):
        self.training_data = training_data
        pass
    
    def predict(self, testing_data):
        predictions = self.training_data.category_id.sample(n=len(testing_data.index), replace=True)
        return predictions
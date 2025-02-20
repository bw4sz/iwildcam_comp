#Random Model to test baseline performance
import numpy as np
class Model():
    
    def __init__(self, config):
        self.config = config
    
    def train(self, generator):
        self.training_data = generator.data
        pass
    
    def predict(self, generator):
        testing_data = generator.data
        predictions = self.training_data.category_id.sample(n=len(testing_data.index), replace=True)
        return predictions
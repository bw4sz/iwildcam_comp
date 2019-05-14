#test utils
#Path hack - how to make this more portable? Not clear to me what root dir is here.
import sys
import pytest

sys.path.append('/Users/ben/Documents/iwildcam_comp/')
from ..DeepTrap import utils
import pandas as pd

def test_voting():
    #read in the sample data input
    prediction_data = pd.read_csv("/Users/ben/Documents/iwildcam_comp/tests/data/sample_prediction.csv")
    desired_output = pd.read_csv("/Users/ben/Documents/iwildcam_comp/tests/data/sample_output.csv")
    
    prediction_data = utils.sequence_voting(prediction_data)
    pd.testing.assert_frame_equal(prediction_data, desired_output)
    
test_voting()

print("test passed")
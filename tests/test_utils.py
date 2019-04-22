from ..DeepTrap import utils
import pytest

def test_read_config():
    config = utils.read_config()
    assert config is not None
    
def test_read_test_data():
    utils.read_test_data()
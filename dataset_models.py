

from data_loaders import load_dataset

from data_transformers import  segment_loaded_data


class hvac_dataset:
    """
    Initialises the data set class using the file name 
    """
    def __init__(self, file_name: str):
        
        self.data = load_dataset(file_name)
        self.segments = segment_loaded_data(self.data)


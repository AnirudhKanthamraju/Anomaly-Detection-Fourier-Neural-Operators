import pandas as pd
from typing import Dict
import warnings


class kernel :
    """
    A kernal class to represent a kernel with its features and labels.
    """
    def __init__(self, features: Dict[str, pd.DataFrame], anomoly: int,  source: str):
        self.features = features  # Dictionary of state space components
        self.label = anomoly      # 0 for normal, 1 for faulty

        self.source = source      # Source file name for traceability
        self.valid =  self.size() is not None 
        
         # Validity check based on consistent DataFrame sizes
    def size(self):
        size_each_state_space_component = {key: len(df) for key, df in self.features.items()}
        
        # Check if all sizes are the same across components
        if len(set(size_each_state_space_component.values())) == 1:
            return size_each_state_space_component[next(iter(size_each_state_space_component))]
        warnings.warn(f"Invalid kernel: Inconsistent DataFrame sizes in features. Sizes: {size_each_state_space_component}", UserWarning)
        return None  # Or handle as needed, e.g., return a default value



import os
import sys
import dill
from src.exception import CustomException

import pandas as pd
import numpy as np

def save_object(filepath,obj):
    
    try:
        dir_path=os.path.dirname(filepath)
        
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath,'wb') as file_object:
            dill.dump(obj,file_object)

    except Exception as e:
        raise CustomException(e,sys)
import os
import sys
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

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
    

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
        
    try:
        report = {}

        for model_name, model in models.items():
            para = params[model_name]

            randomized_search = RandomizedSearchCV(model,para,cv=5,verbose=2)
            randomized_search.fit(X_train,y_train)

            model.set_params(**randomized_search.best_params_)

            model.fit(X_train,y_train)


            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_model_score = r2_score(y_train,y_pred_train)
            test_model_score = r2_score(y_test,y_pred_test)

            report[model_name]=test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)
    

import sys
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

from src.utils import save_object
from src.utils import evaluate_models

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self,train_data,test_data):
        
        try:
            logging.info("splitting training and test input data")

            X_train,y_train,X_test,y_test = (

                train_data[:,:-1],
                train_data[:,-1],
                test_data[:,:-1],
                test_data[:,-1]
            )

            models = {
                        "LinearRegression": LinearRegression(),
                        "SVR": SVR(),
                        "KNeighborsRegressor": KNeighborsRegressor(),
                        "DecisionTreeRegressor": DecisionTreeRegressor(),
                        "RandomForestRegressor": RandomForestRegressor(),
                        "AdaBoostRegressor": AdaBoostRegressor(),
                        "GradientBoostingRegressor": GradientBoostingRegressor(),
                        "XGBRegressor": XGBRegressor(),
                        "CatBoostRegressor": CatBoostRegressor(verbose=0)  # Suppress verbose output
                    }


            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,
                                        X_test=X_test,y_test=y_test,models=models) ## utils
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = best_model_name = max(model_report, key=model_report.get)

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best models found")
            logging.info("Best  model found on both training and testing dataset")

            save_object(

                filepath=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved in pickle file")

            predicted = best_model.predict(X_test)
            r_score = r2_score(y_test,predicted)
            return best_model_name,r_score

        except Exception as e:
            raise CustomException(e,sys)
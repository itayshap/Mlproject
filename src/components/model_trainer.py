import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_mode_file_path = os.path.join("artifacts" ,"model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    def initate_model_trainer(self, train_array, test_array):
        try:    
            logging.info("Split train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],               
                ) 
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }   

            model_report : dict = evaluate_models(X_train, y_train, X_test, y_test, 
                                                 models=models)   
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")   

            save_object(self.model_trainer_config.trained_mode_file_path, best_model)
            return best_model_score
        except Exception as e:
            raise CustomException(e, sys.exc_info()[2])

if __name__ == '__main__':
    x= DataIngestion()
    train_path, test_path = x.initiate_data_ingestion()
    y= DataTransformation("math_score")
    train, test, path = y.initiate_data_transformation(train_path, test_path)
    z = ModelTrainer()
    best_score = z.initate_model_trainer(train, test)
    print(best_score)



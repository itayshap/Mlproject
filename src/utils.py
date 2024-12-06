import os
import sys
import numpy as np 
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys.exc_info()[2])
    
def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, models:dict, params:dict):
    try:
        models_report = {}
        best_model = None
        best_model_score = 0 
        for name, model in models.items():
            model_params = params[name]
            gs = GridSearchCV(model, model_params,cv=3, verbose=False,)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            # Make predictions
            y_test_pred = model.predict(X_test)
            
            # Evaluate Test dataset
            model_test_r2 = r2_score(y_test, y_test_pred)
            models_report[name] = model_test_r2
            if model_test_r2 > best_model_score:
                best_model_score = model_test_r2
                best_model = model
        return best_model_score, best_model
    except Exception as e:
        raise CustomException(e, sys.exc_info()[2])
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys.exc_info()[2])

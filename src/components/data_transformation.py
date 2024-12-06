from src.components.data_ingestion import DataIngestion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os
import sys
from src.utils import save_object
from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self, target_column: str) -> None:
        self.data_transformation_config = DataTransformationConfig()
        self.target_column = target_column
    
    def get_features(self, train_set: pd.DataFrame) -> tuple[list, list]:
        num_features = train_set.select_dtypes(include='number').columns
        num_features = num_features.drop(labels=self.target_column, errors='ignore').to_list()
        cat_features = train_set.select_dtypes(include='object').columns
        cat_features = cat_features.drop(labels=self.target_column, errors='ignore').to_list()
        return num_features, cat_features
 
    def get_data_tranformer_object(self, cat_features: list, num_features: list):

        '''
        This function creates and return data transformer
        '''

        try:
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )
            logging.info(f"Categorical columns:{cat_features}")
            logging.info(f"Numerical columns:{num_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys.exc_info()[2])
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            num_features, cat_features = self.get_features(train_df)
            preprocessing_obj = self.get_data_tranformer_object(cat_features, num_features)

            logging.info("Obtaining preprocessing object")

            X_train=train_df.drop(columns=[self.target_column],axis=1)
            y_train=train_df[self.target_column]

            X_test=test_df.drop(columns=[self.target_column],axis=1)
            y_test=test_df[self.target_column]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            X_train_transformed=preprocessing_obj.fit_transform(X_train)
            X_test_transformed=preprocessing_obj.transform(X_test)


            logging.info(f"Saved preprocessing object.")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)
            
            train_set = np.c_[
                X_train_transformed, np.array(y_train)
            ]
            test_set = np.c_[X_test_transformed, np.array(y_test)]

            return (train_set ,test_set, self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e,sys.exc_info()[2])
        

if __name__ == "__main__":
    x= DataIngestion()
    train_path, test_path = x.initiate_data_ingestion()
    y= DataTransformation("math_score")
    train, test, path = y.initiate_data_transformation(train_path, test_path)
import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

FILE_NAME = 'boston.csv'
TEST_FILE = "test_data.csv"

MODEL_NAME = 'classification.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')



TARGET = 'MEDV'

#Final features used in the model
FEATURES = ['CRIM', 'ZN', 'INDUS', 'CHAS',
       'NOX', 'RM', 'AGE', 'DIS',
       'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']

PRED_FEATURES = ['CRIM', 'ZN', 'INDUS', 'CHAS',
       'NOX', 'RM', 'AGE', 'DIS',
       'RAD', 'TAX', 'PTRATIO','B', 'LSTAT']

NUM_FEATURES = ['CRIM', 'ZN', 'INDUS', 'CHAS',
       'NOX', 'RM', 'AGE', 'DIS',
       'RAD', 'TAX', 'PTRATIO','B', 'LSTAT']

CAT_FEATURES = []







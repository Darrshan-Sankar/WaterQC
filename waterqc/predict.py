from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import polars as pl
import pandas as pd
import joblib
import os
import pathlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

def predict():
    print("Enter informations for prediction one by one for each criteria: ")
    cols = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color', 'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity', 'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Source', 'Water Temperature', 'Air Temperature', 'Month', 'Day', 'Time of Day']
    req_col = ['Color', 'Source', 'Month']
    non_req = ['Month', 'Day', 'Source', 'Air Temperature', 'Conductivity', 'Time of Day']
    test_data = []
    for i in cols:
        if i in req_col:
            if i not in non_req:
                coder = le()
                coder.classes_ = np.load(pathlib.Path(__file__).parent.joinpath("Color.npy"), allow_pickle=True)
                print("DONE")        
                try:
                    print("The available value for color is: ['Colorless','Faint Yellow','Light Yellow','Near Colorless','Yellow']")
                    print("Input any one of the above available colors......")
                    sample = input("Enter value for "+str(i)+": ")
                    available = ['Colorless','Faint Yellow','Light Yellow','Near Colorless','Yellow']
                    if sample not in available:
                        raise Exception("Color is invalid....")
                    sample = coder.transform(sample)
                except ValueError:
                    sample = None
                test_data.append(sample)
            else:
                test_data.append(None)
        elif i in non_req:
            sample = None
            test_data.append(sample)
        else:
            try:
                sample = float(input("Enter value for "+str(i)+": "))
            except ValueError:
                sample = None
            test_data.append(sample)
    scaler = joblib.load(pathlib.Path(__file__).parent.joinpath("scaler.save"))
    for i in os.listdir(pathlib.Path(__file__).parent):
        if i.startswith("model_"):
            saved_model = joblib.load(pathlib.Path(__file__).parent.joinpath(i))
            print(".....Loading model......")
    test_data = scaler.transform(pd.DataFrame(test_data).T)
    test_data = pd.DataFrame(test_data)
    test_data.columns = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color', 'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity', 'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Source', 'Water Temperature', 'Air Temperature', 'Month', 'Day', 'Time of Day']
    test_data.drop(['Month', 'Day', 'Source', 'Air Temperature', 'Conductivity', 'Time of Day'], axis = 1, inplace=True)
    test_data = pd.DataFrame(test_data)
    test_pred = saved_model.predict(test_data)
    print(test_pred)
    print(test_pred[0])
    if test_pred[0]==0:
        result = "Water quality is bad!!"
    else:
        result = "Water quality is good!!"
    return(result)

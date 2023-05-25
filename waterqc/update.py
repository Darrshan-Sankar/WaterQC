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

def update():
    cols = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color', 'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity', 'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Source', 'Water Temperature', 'Air Temperature', 'Month', 'Day', 'Time of Day']
    req_col = ['Color', 'Source', 'Month']
    non_req = ['Month', 'Day', 'Source', 'Air Temperature', 'Conductivity', 'Time of Day']
    data = pl.read_csv(pathlib.Path(__file__).parent.joinpath("dataset.csv")).to_pandas()
    test_data = []
    print("...The existing dataset contains columns in following order.\nPlease follow the order to update the data.\nEnter data with appropriate range and values!...")
    print("'Index','pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color', 'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity', 'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Source', 'Water Temperature', 'Air Temperature', 'Month', 'Day', 'Time of Day'")
    dat = int(input("\nEnter the format of data to update: '1' for single record and '2' for adding another dataset with same columns: "))
    length = int(len(data.iloc[ : ,1]))
    if dat ==1:
        print("Enter informations for appending one by one for each criteria: ")
        test_data.append(length)
        coder = le()
        coder.classes_ = np.load(pathlib.Path(__file__).parent.joinpath("Color.npy"), allow_pickle=True)
        for i in cols:
            if i in req_col:
                if i not in non_req:        
                    try:
                        print("The available value for color is: ['Colorless','Faint Yellow','Light Yellow','Near Colorless','Yellow']")
                        print("Input any one of the above available colors......")
                        sample = input("Enter value for "+str(i)+": ")
                        available = ['Colorless','Faint Yellow','Light Yellow','Near Colorless','Yellow']
                        if sample not in available:
                            raise Exception("Color is invalid....")
                        sample = sample
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
        target = int(input("Enter the input for target '0' for 'bad quality' and '1' for 'good quality': "))
        test_data.append(target)
        len(test_data)
        test_data = pd.DataFrame(test_data).T
        test_data.columns = ['Index','pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color', 'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity', 'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Source', 'Water Temperature', 'Air Temperature', 'Month', 'Day', 'Time of Day', 'Target']
        test_data = pd.DataFrame(test_data)
        final = data.append(test_data)
        del data
        del test_data
        final = pl.DataFrame(final)
        final.write_csv(pathlib.Path(__file__).parent.joinpath("dataset.csv"))
        print("Dataset UPDATED!!")
    else:
        coder = le()
        coder.classes_ = np.load(pathlib.Path(__file__).parent.joinpath("Color.npy"), allow_pickle=True)
        loc = input("Enter the location of dataset without quotes: ")
        print(loc)
        loc = pathlib.Path(loc)
        test_data = pl.read_csv(loc).to_pandas()
        test_data.Index = length+test_data.Index
        final = data.append(test_data)
        del data
        del test_data
        final = pl.DataFrame(final)
        final.write_csv(pathlib.Path(__file__).parent.joinpath("dataset.csv"))
        print("Dataset UPDATED!!")
    return final

def update_retrain():
    cols = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color', 'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity', 'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Source', 'Water Temperature', 'Air Temperature', 'Month', 'Day', 'Time of Day']
    req_col = ['Color', 'Source', 'Month']
    non_req = ['Month', 'Day', 'Source', 'Air Temperature', 'Conductivity', 'Time of Day']
    data = pl.read_csv(pathlib.Path(__file__).parent.joinpath("dataset.csv")).to_pandas()
    test_data = []
    print("...The existing dataset contains columns in following order.\nPlease follow the order to update the data.\nEnter data with appropriate range and values!...")
    print("'Index','pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color', 'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity', 'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Source', 'Water Temperature', 'Air Temperature', 'Month', 'Day', 'Time of Day'")
    dat = int(input("\nEnter the format of data to update: '1' for single record and '2' for adding another dataset with same columns: "))
    length = int(len(data.iloc[ : ,1]))
    if dat ==1:
        print("Enter informations for appending one by one for each criteria: ")
        test_data.append(length)
        coder = le()
        coder.classes_ = np.load(pathlib.Path(__file__).parent.joinpath("Color.npy"), allow_pickle=True)
        for i in cols:
            if i in req_col:
                if i not in non_req:        
                    try:
                        print("The available value for color is: ['Colorless','Faint Yellow','Light Yellow','Near Colorless','Yellow']")
                        print("Input any one of the above available colors......")
                        sample = input("Enter value for "+str(i)+": ")
                        available = ['Colorless','Faint Yellow','Light Yellow','Near Colorless','Yellow']
                        if sample not in available:
                            raise Exception("Color is invalid....")
                        sample = sample
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
        target = int(input("Enter the input for target '0' for 'bad quality' and '1' for 'good quality': "))
        test_data.append(target)
        len(test_data)
        test_data = pd.DataFrame(test_data).T
        test_data.columns = ['Index','pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color', 'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity', 'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Source', 'Water Temperature', 'Air Temperature', 'Month', 'Day', 'Time of Day', 'Target']
        test_data = pd.DataFrame(test_data)
        final = data.append(test_data)
        del data
        del test_data
        final = pl.DataFrame(final)
        final.write_csv(pathlib.Path(__file__).parent.joinpath("dataset.csv"))
        print("Dataset UPDATED!!")
    else:
        coder = le()
        coder.classes_ = np.load(pathlib.Path(__file__).parent.joinpath("Color.npy"), allow_pickle=True)
        loc = input("Enter the location of dataset without quotes: ")
        print(loc)
        loc = pathlib.Path(loc)
        test_data = pl.read_csv(loc).to_pandas()
        test_data.Index = length+test_data.Index
        final = data.append(test_data)
        del data
        del test_data
        final = pl.DataFrame(final)
        final.write_csv(pathlib.Path(__file__).parent.joinpath("dataset.csv"))
        print("Dataset UPDATED!!")
    scaler = joblib.load(pathlib.Path(__file__).parent.joinpath("scaler.save"))
    for i in os.listdir(pathlib.Path(__file__).parent):
        if i.startswith("model_"):
            print(".....Loading old model......")
            saved_model = joblib.load(pathlib.Path(__file__).parent.joinpath(i))
            old_model = pathlib.Path(__file__).parent.joinpath(i)
            old_accuracy = i[:-4].split('_')[1]
            print("The old model's accuracy is: ", old_accuracy)
    print("Started Preprocessing")
    del final
    final = pl.read_csv(pathlib.Path(__file__).parent.joinpath("dataset.csv")).to_pandas()
    final.dropna(inplace = True)
    X = final.drop(['Target', 'Index'],axis = 1)
    y = final.Target
    del final
    X.Color = coder.transform(X.Color)
    coder.classes_ = np.load(pathlib.Path(__file__).parent.joinpath("Source.npy"), allow_pickle=True)
    X.Source = coder.transform(X.Source)
    coder.classes_ = np.load(pathlib.Path(__file__).parent.joinpath("Month.npy"), allow_pickle=True)
    X.Month = coder.transform(X.Month)
    print("Scaling")
    X = scaler.transform(X)
    X = pd.DataFrame(X)
    X.columns = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color', 'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity', 'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Source', 'Water Temperature', 'Air Temperature', 'Month', 'Day', 'Time of Day']
    print("Scaled")
    X = X.drop(['Month', 'Day', 'Source', 'Air Temperature', 'Conductivity', 'Time of Day'], axis = 1)
    print("Preprocessed")
    print(".....Training new model.....")
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    model = XGBClassifier(n_estimators= 16, learning_rate=1, objective='binary:logistic').fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    new_accuracy = accuracy_score(Y_test, y_pred)
    if float(old_accuracy)>new_accuracy:
        return("New model's accuracy is: "+str(new_accuracy)+". Old model is better than new model. No changes made")
    else:
        os.delete(old_model)
        joblib.dump(pathlib.Path(__file__).parent.joinpath("model_"+str(new_accuracy)+".pkl"))
        return("New model's accuracy is: "+str(new_accuracy)+"\n\n....Replaced Old model with new model")

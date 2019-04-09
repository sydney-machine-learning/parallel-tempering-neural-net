import sklearn.preprocessing as preprocess
import numpy as np
import pandas as pd
 


def load_data(filename):
    df = pd.read_csv(filename, sep=";")
    cols_numerical = list(df.columns[(df.dtypes=="int")|(df.dtypes=="float")])
    cols_categorical = list(df.columns[df.dtypes=="object"].drop('y'))

    # Output labels (success/failure)
    y = pd.get_dummies(df["y"])["yes"].values.astype("float32")
    # Numerical inputs
    X = df[cols_numerical]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # Categorical inputs
    for name in cols_categorical:
        X = pd.concat((X, pd.get_dummies(df[name])), axis=1)
    return X.values.astype("float32"), np.reshape(y, (-1,1))

X, y = load_data("./bank.csv")
print("X:{}, y:{}".format(X.shape, y.shape))
np.savetxt('bank-processed.csv',np.hstack((X,y)),delimiter=';')

import numpy as np
import os
import pickle
import pandas as pd

def train_test_split(X,y,test_size=0.3,train_size=None,random_state = None):
    if random_state is not None:
        np.random.seed(seed = random_state)
    if train_size:
        test_size = 1.0 - train_size
    X = np.array(X)
    y = np.array(y)
    size = X.shape[0]
    indx = np.random.choice(size,int(size * test_size),replace=False)
    mask = ~np.isin(range(size),indx)
    return X[mask],X[indx],y[mask],y[indx]

def accuracy_score(data1,data2):
    return np.mean(np.array(data1) == np.array(data2))

def save_model(model,model_path):
    if not os.path.exists("Saved_Model/"):
        os.makedirs('Saved_Model/')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

def load_model(pkl_filename):
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model

def pandas_to_csv(file,filename):
    file.to_csv(filename,index=False)

def get_data(path,num_rows):
    try:
        print("Got Data...")
        return pd.read_csv(path, delimiter='\t', header=None).head(num_rows)
    except Exception as error:
        print(error)
        return None
import config
import os
import model
import utils
import time
import numpy as np
import pandas as pd
from sklearn import svm 
import warnings

def train_model():
    Model = model.BertForFakeNewsDetection()
    features, labels = Model.get_features(train=True)
    X_train,X_test,y_train,y_test = utils.train_test_split(features,labels,test_size=0.3)
    clf = svm.SVC()
    clf.fit(features,labels)
    print("Validation Accuracy:",round(clf.score(X_test,y_test),4) * 100,"%")
    clf.fit(X_test,y_test)
    utils.save_model(clf,config.MODEL_PATH)

def save_submission(results):
    submission = pd.read_csv(config.SUBMISSION_PATH)
    submission['Category'] = results
    file_name = config.SUBMISSION_PATH
    utils.pandas_to_csv(submission,file_name)

def predict():
    Model = model.BertForFakeNewsDetection()
    features = Model.get_features(train=False)
    clf = utils.load_model(config.MODEL_PATH)
    results = clf.predict(features)
    save_submission(results)
    


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_model()
    predict()
        
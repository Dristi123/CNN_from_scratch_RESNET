import csv
import sys

import numpy as np
import pickle
import glob
import cv2
import pandas as pd
import os
import csv
from train_1705029 import *
def preprocess(set_X):
    X = 255 - set_X
    X = np.where(X < 80, 0, 255)
    X = X / 255
    # mean=np.mean(X)
    # std=np.std(X)
    # X=(X-mean)/std
    return X


global filenames
def load_test_data(path,flag):
    cv_img_2=os.listdir(path)
    #print(cv_img_2)
    filenames=[]
    test_set_X=[]
    #filenames=cv_img_2
    #test_set_X = np.array([cv2.resize(cv2.imread((os.path.join(path,f))),(32, 32),interpolation=cv2.INTER_LINEAR) for f in cv_img_2])
    for filename in os.listdir(path):
        if ("png" in filename):
            #global  filenames
            filenames.append(filename)
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                test_set_X.append(cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR))
    #return np.array(images), file_list
    test_set_X = preprocess(np.array(test_set_X))
    test_set_Y=None
    if flag==True:
        df = pd.read_csv("training-d.csv", usecols=['digit'])
        test_set_Y = df.to_numpy()
        test_set_Y = test_set_Y.reshape(-1)
    return filenames,test_set_X,test_set_Y


if __name__ == '__main__':
    path=sys.argv[1]
    print(path)
    path_2=str(path)+"\\1705029_prediction.csv"
    #print(path_2)
    filenames,test_x,test_y=load_test_data(path,False)
    file_name="1705029_model.pickle"
    with open(file_name, 'rb') as file:
        loaded_model = pickle.load(file)
    y_pred=loaded_model.test(test_x,test_y,False)
    with open(path_2,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['FileName','Digit'])
        for i in range(len(filenames)):
            w.writerow([filenames[i],y_pred[i]])
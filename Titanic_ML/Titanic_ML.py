# Titanic ML
# https://www.kaggle.com/c/titanic/team
# TeeJ Lockwood

# Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Sklearn Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.externals import joblib

################################################
############        Data Explore    ############
################################################
def explore_data(inpath=["None.csv"]):
    # Explore the data for null values and outliers

    # Pull Data
    if inpath[0] == "None.csv": return # stop script if no file specified
    for path in inpath:
        print ("Reading in the file path: " + path, ' ')
        df = pd.read_csv(path) # read in the file path
        print ("Showing off the tail and the head.", ' ')
        print("Head")
        print(df.head()) # show the head of each data set
        print("Tail")
        print(df.tail()) # show the tail of each data set

explore_data(inpath = [os.path.join('Raw Data', path) for path in os.listdir('Raw Data')])

################################################
############        Run Models      ############
################################################
def run_model(inpath = "None.csv"):
    # Pull Data
    if inpath == "None.csv": return # stop script if no file specified
    df = read_data(inpath) # get input
    feat = df.iloc[:,:-1] # get features
    label = df.iloc[:,-1] # get labels

    # Show Data
    # print(feat.head()) # print(label.head())
    #eda(df) # plot distribution by variable
    
    # Feature Selection - Train Test Split
    X_train, X_test, y_train, y_test = feature_selection(feat, label)

    # Train Model
    dtree = learn_tree(X_train,y_train)

    # Test Model
    predictions = test_model(dtree, X_test, y_test)
    plot_test(y_test, predictions)

    # Save Model
    export_model(dtree, 'dtree_stat_win.pk1')
    

################################################
############        Tool Box        ############
################################################

def read_data(in_path):
    # Establish filepath
    #in_path = "sample_data_num.csv"
    # Read in data
    return (pd.read_csv(in_path, sep="|"))

def eda(df):
    # plot variable distribution, still confused on how it works
    g = sns.pairplot(df)
    plt.show()

def feature_selection(X, y):
    # import feature selection package
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.30)

def learn_tree(X_train,y_train):
    # import decision trees and train model
    from sklearn.tree import DecisionTreeClassifier
    dtree = DecisionTreeClassifier()
    return dtree.fit(X_train,y_train)

def test_model(model, X_test, y_test):
    # import model testing
    from sklearn.metrics import classification_report,confusion_matrix
    predictions = model.predict(X_test)
    print(classification_report(y_test,predictions)) # show results
    print(confusion_matrix(y_test,predictions)) # show results
    return predictions

def export_model(model, out_path):
    # import joblib
    from sklearn.externals import joblib
    joblib.dump(model, out_path) #output model

def import_model(model, in_path):
    # import joblib
    from sklearn.externals import joblib
    return joblib.load(in_path)

def plot_test(y_test, predictions):
    sns.distplot((y_test-predictions))
    plt.show()
    

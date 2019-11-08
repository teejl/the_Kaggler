# ML DEMO - Decision Trees
# 80053507
# 2019-09-03

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Pull Data
    print("Pulling Data...")
    df = read_data("train.csv") # get input
    
    # Mettle with data really quick
    print("Mettling with Data...")
    df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = df.dropna()

    # Split data into features and labels
    feat = df[[col for col in df.columns if col != "Survived"]] # get features
    label = df["Survived"] # get labels

    # Show Data
    print("Sample Data:")
    print(feat.head())
    print(label.head())
    #eda(df) # plot distribution by variable

    # Feature Selection - Train Test Split
    print("Splitting Data...")
    X_train, X_test, y_train, y_test = feature_selection(feat, label)

    # Train Models
    print("Training Models...")
    dtree = learn_tree(X_train,y_train)
    KNN = learn_KNN(2,X_train, y_train)

    for model in [dtree, KNN]:
        # Test Model
        print("Testing Models...")
        print(model)
        predictions = test_model(model, X_test, y_test)
        #plot_test(y_test, predictions)
        
    # Save Model
    print("Saving Models...")
    export_model(dtree, 'dtree_titanic.pk1')
    export_model(KNN, 'KNN_titanic.pk1')

    # Sign Off
    print("Complete!")

def read_data(in_path):
    # Establish filepath
    #in_path = "sample_data_num.csv"
    # Read in data
    return (pd.read_csv(in_path))

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

def learn_KNN(K, X_train, y_train):
    # import KNN modeling
    from sklearn.neighbors import KNeighborsClassifier
    KNN = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)
    return KNN

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

# RUN CODE HERE
for i in range(10):
    main()

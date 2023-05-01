#Name: Chris Forte
#Date: 30/04/2023
#Course: COMPSCI 711
#Task: Assignment III

#This program performs various functions, which are organized into four sections.

#The first section features several functions that support a 10-fold cross-validation on a classification
#dataset via three different types of neural networks. These are: one having a hidden layer with very few
#nodes, one having a hidden layer with a reasonable number of nodes, and one having a hidden layer with
#too many nodes.

#The second section features a function titled "classification_task()", which compares the accuracy results
#generated from the neural networks used to examine the classification dataset.

#The third section features several functions that support a 10-fold cross-validation on a regression
#dataset via three different types of neural networks. These are: one having a hidden layer with very few
#nodes, one having a hidden layer with a reasonable number of nodes, and one having a hidden layer with
#too many nodes.

#The fourth section features a function titled "regression_task()", which compares the rmse results
#generated from the neural networks used to examine the regression dataset.

#Referenced resources and tools include https://www.geeksforgeeks.org, https://www.openml.org,
#"Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples,
#and Case Studies", https://docs.python.org/, and https://chat.openai.com. 


import numpy as np
import openml
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from tensorflow.keras import models,layers
import plotly.graph_objects as go



#Classification Cross Validation (Too Few Nodes)
def classification_few():
    wine = datasets.fetch_openml(data_id=44091)

    hot = OneHotEncoder(sparse=False)
    lst = [[x] for x in wine.target]
    num = hot.fit_transform(lst)

    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    test_lst = []

    for train, test in kfold.split(wine.data, num) :
        ntwrk = models.Sequential()
        ntwrk.add(layers.Dense(1, activation="relu", input_dim=11))
        ntwrk.add(layers.Dense(2, activation="softmax"))
        ntwrk.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        
        ntwrk_epochs = 50
        ntwrk.fit(wine.data.iloc[train], num[train], epochs=ntwrk_epochs)

        test_det = ntwrk.evaluate(wine.data.iloc[test], num[test])
        test_lst.append(test_det[1])

        print("Fold",len(test_lst), "Test Accuracy =", test_det[1])

    return test_lst, np.mean(test_lst)


#Classification Cross Validation (Just Right Nodes)
def classification_right():
    wine = datasets.fetch_openml(data_id=44091)

    hot = OneHotEncoder(sparse=False)
    lst = [[x] for x in wine.target]
    num = hot.fit_transform(lst)

    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    test_lst = []

    for train, test in kfold.split(wine.data, num) :
        ntwrk = models.Sequential()
        ntwrk.add(layers.Dense(20, activation="relu", input_dim=11))
        ntwrk.add(layers.Dense(2, activation="softmax"))
        ntwrk.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        
        ntwrk_epochs = 25
        ntwrk.fit(wine.data.iloc[train], num[train], epochs=ntwrk_epochs)

        test_det = ntwrk.evaluate(wine.data.iloc[test], num[test])
        test_lst.append(test_det[1])

        print("Fold",len(test_lst), "Test Accuracy =", test_det[1])

    return test_lst, np.mean(test_lst)


#Classification Cross Validation (Too Many Nodes)
def classification_many():
    wine = datasets.fetch_openml(data_id=44091)

    hot = OneHotEncoder(sparse=False)
    lst = [[x] for x in wine.target]
    num = hot.fit_transform(lst)

    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    test_lst = []

    for train, test in kfold.split(wine.data, num) :
        ntwrk = models.Sequential()
        ntwrk.add(layers.Dense(2000, activation="relu", input_dim=11))
        ntwrk.add(layers.Dense(2, activation="softmax"))
        ntwrk.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        
        ntwrk_epochs = 5
        ntwrk.fit(wine.data.iloc[train], num[train], epochs=ntwrk_epochs)

        test_det = ntwrk.evaluate(wine.data.iloc[test], num[test])
        test_lst.append(test_det[1])

        print("Fold",len(test_lst), "Test Accuracy =", test_det[1])

    return test_lst, np.mean(test_lst)


#classification Assignment Task I
def classification_task():

    test_lst_few, avg_acc_few = classification_few()
    test_lst_right, avg_acc_right = classification_right()
    test_lst_many, avg_acc_many = classification_many()
    
    conc_acc_few = [f"{acc:.4f}" for acc in test_lst_few] + [f"<b>{avg_acc_few:.4f}</b>"]
    conc_acc_right = [f"{acc:.4f}" for acc in test_lst_right] + [f"<b>{avg_acc_right:.4f}</b>"]
    conc_acc_many = [f"{acc:.4f}" for acc in test_lst_many] + [f"<b>{avg_acc_many:.4f}</b>"]

    fig = go.Figure(data=[go.Table(
        header=dict(values=['Accuracy Rates', 'Few Nodes', 'Reasonable Nodes', 'Many Nodes']),
        cells=dict(values=[['Fold-Acc 1', 'Fold-Acc 2', 'Fold-Acc 3', 'Fold-Acc 4', 
                            'Fold-Acc 5', 'Fold-Acc 6', 'Fold-Acc 7', 'Fold-Acc 8', 
                            'Fold-Acc 9', 'Fold-Acc 10', 'Fold Average'],
                            conc_acc_few,
                            conc_acc_right,
                            conc_acc_many])
    )])
    
    fig.update_layout(title='Classification 10-Fold Cross-Validation Neural Network Comparison',
                      font=dict(size=12),
                      height=400,
                      margin=dict(l=50, r=50, t=50, b=50))
    fig.show()   



#Regression Cross Validation (Too Few Nodes)
def regression_few():
    wine = datasets.fetch_openml(data_id=287)

    X = wine.data.values
    Y = wine.target.values

    hot = OneHotEncoder(sparse=False)
    num = hot.fit_transform(Y.reshape(-1,1))

    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    test_lst = []

    for train, test in kfold.split(X, num):
        x_train = X[train]
        y_train = num[train]
        x_test = X[test]
        y_test = num[test]
        
        ntwrk = models.Sequential()
        ntwrk.add(layers.Dense(1, activation="relu", input_dim=11))
        ntwrk.add(layers.Dense(1))
        ntwrk.compile(optimizer="rmsprop", loss="mse", metrics=["mae","mse"])
        
        ntwrk_epochs = 2
        history = ntwrk.fit(x_train, y_train, epochs=ntwrk_epochs)

        test_det = ntwrk.evaluate(x_test, y_test)
        test_lst.append(test_det[1])

        print("Fold",len(test_lst), "Test RMSE =", test_det[1])

    return test_lst, np.mean(test_lst)


#Regression Cross Validation (Just Right Nodes)
def regression_right():
    wine = datasets.fetch_openml(data_id=287)

    X = wine.data.values
    Y = wine.target.values

    hot = OneHotEncoder(sparse=False)
    num = hot.fit_transform(Y.reshape(-1,1))

    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    test_lst = []

    for train, test in kfold.split(X, num):
        x_train = X[train]
        y_train = num[train]
        x_test = X[test]
        y_test = num[test]
        
        ntwrk = models.Sequential()
        ntwrk.add(layers.Dense(30, activation="relu", input_dim=11))
        ntwrk.add(layers.Dense(1))
        ntwrk.compile(optimizer="rmsprop", loss="mse", metrics=["mae","mse"])
        
        ntwrk_epochs = 2
        history = ntwrk.fit(x_train, y_train, epochs=ntwrk_epochs)

        test_det = ntwrk.evaluate(x_test, y_test)
        test_lst.append(test_det[1])

        print("Fold",len(test_lst), "Test RMSE =", test_det[1])

    return test_lst, np.mean(test_lst)


#Regression Cross Validation (Too Many Nodes)
def regression_many():
    wine = datasets.fetch_openml(data_id=287)

    X = wine.data.values
    Y = wine.target.values

    hot = OneHotEncoder(sparse=False)
    num = hot.fit_transform(Y.reshape(-1,1))

    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    test_lst = []

    for train, test in kfold.split(X, num):
        x_train = X[train]
        y_train = num[train]
        x_test = X[test]
        y_test = num[test]
        
        ntwrk = models.Sequential()
        ntwrk.add(layers.Dense(1000, activation="relu", input_dim=11))
        ntwrk.add(layers.Dense(1))
        ntwrk.compile(optimizer="rmsprop", loss="mse", metrics=["mae","mse"])
        
        ntwrk_epochs = 8
        history = ntwrk.fit(x_train, y_train, epochs=ntwrk_epochs)

        test_det = ntwrk.evaluate(x_test, y_test)
        test_lst.append(test_det[1])

        print("Fold",len(test_lst), "Test RMSE =", test_det[1])

    return test_lst, np.mean(test_lst)


#Regression Assignment Task II
def regression_task():

    test_lst_few, avg_rmse_few = regression_few()
    test_lst_right, avg_rmse_right = regression_right()
    test_lst_many, avg_rmse_many = regression_many()
    
    conc_rmse_few = [f"{rmse:.4f}" for rmse in test_lst_few] + [f"<b>{avg_rmse_few:.4f}</b>"]
    conc_rmse_right = [f"{rmse:.4f}" for rmse in test_lst_right] + [f"<b>{avg_rmse_right:.4f}</b>"]
    conc_rmse_many = [f"{rmse:.4f}" for rmse in test_lst_many] + [f"<b>{avg_rmse_many:.4f}</b>"]

    fig = go.Figure(data=[go.Table(
        header=dict(values=['RMSE Rates', 'Few Nodes', 'Reasonable Nodes', 'Many Nodes']),
        cells=dict(values=[['Fold-RMSE 1', 'Fold-RMSE 2', 'Fold-RMSE 3', 'Fold-RMSE 4', 
                            'Fold-RMSE 5', 'Fold-RMSE 6', 'Fold-RMSE 7', 'Fold-RMSE 8', 
                            'Fold-RMSE 9', 'Fold-RMSE 10', 'Fold Average'],
                            conc_rmse_few,
                            conc_rmse_right,
                            conc_rmse_many])
    )])
    
    fig.update_layout(title='Regression 10-Fold Cross-Validation Neural Network Comparison',
                      font=dict(size=12),
                      height=400,
                      margin=dict(l=50, r=50, t=50, b=50))
    fig.show()   

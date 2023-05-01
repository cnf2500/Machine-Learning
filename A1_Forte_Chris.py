#Name: Chris Forte
#Date: 04/03/2023
#Course: COMPSCI 711
#Task: Assignment I

#This program features two functions.

#The first is decision_tree_parameter_check(dataset_id), which takes an OpenML
#dataset identification number as input and produces a graph as output. This
#function varies the values of 5 min_samples_leaf parameters and measures
#training and test roc_auc scores on a 10-fold cross-validation.

#The second is decision_tree_parameter_gridsearch(dataset_id), which takes an
#OpenML dataset identification number as input and produces a graph and numerical
#value as output.This function uses GridSearchCV to search for the best parameter
#and generate the results of 10-fold cross-validation.

#Referenced resources include https://www.geeksforgeeks.org, https://www.openml.org,
#"Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked 
#Examples, and Case Studies", and https://docs.python.org/. 


import openml
import matplotlib.pyplot as mpl

from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


def decision_tree_parameter_check(dataset_id):
    #Load OpenML dataset.
    dst = openml.datasets.get_dataset(dataset_id)

    #Get the input features and target variable as NumPy arrays.
    data = dst.get_data(target=dst.default_target_attribute)

    #Store the input features and target variable.
    X = data[0]
    y = data[1]

    #List of values to vary min_samples_leaf
    min_samples_leaf_values = [1, 3, 5, 7, 9]

    #Empty lists to store mean training and test roc_auc scores.
    training = []
    testing = []

    #Loop through the values of min_samples_leaf.
    for min_samples_leaf in min_samples_leaf_values:

        #Create a DecisionTreeClassifier via min_samples_leaf.
        mytree = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=min_samples_leaf)

        #Calculate the mean roc_auc score on a 10-fold cross-validation.
        train_cv = cross_val_score(mytree, X, y, cv=10, scoring='roc_auc').mean()
        training.append(train_cv)

        #Fit the model on the entire dataset. 
        mytree.fit(X, y)
        y_prob = mytree.predict_proba(X)

        #Calculate the test roc_auc score.
        test_cv = roc_auc_score(y, y_prob[:, 1])
        testing.append(test_cv)

    #Plot results on graph.
    mpl.plot(min_samples_leaf_values, training, label='train')
    mpl.plot(min_samples_leaf_values, testing, label='test')
    mpl.xlabel('min_samples_leaf')
    mpl.ylabel('roc_auc_score')
    mpl.title(f"DTC Dataset ID: {dataset_id} Graph")
    mpl.legend()

    #Shading for underfitting.
    mpl.fill_between(min_samples_leaf_values, 0, training, alpha=0.1, color='blue')
    mpl.text(min_samples_leaf_values[0], 0.6, 'Underfitting', rotation=90, color='blue', ha='center', va='center')

    #Shading for overfitting.
    mpl.fill_between(min_samples_leaf_values, testing, 1, alpha=0.1, color='red')
    mpl.text(min_samples_leaf_values[0], 0.9, 'Overfitting', rotation=90, color='red', ha='center', va='center')

    #Return graph.
    mpl.show()


def decision_tree_parameter_gridsearch(dataset_id):
    #Load OpenML dataset.
    dst = openml.datasets.get_dataset(dataset_id)

    #Get the input features and target variable as NumPy arrays.
    data = dst.get_data(target=dst.default_target_attribute)

    #Store the input features and target variable.
    X = data[0]
    y = data[1]

    #Declare mytree as DecisionTreeClassifier.
    mytree = tree.DecisionTreeClassifier(criterion="entropy")

    #Dictionary of parameters to search through.
    parameters = {'min_samples_leaf': [1, 3, 5, 7, 9]}

    #Grid search using GridSearchCV.
    gs = GridSearchCV(mytree, param_grid=parameters, cv=10, scoring='roc_auc')
    gs.fit(X, y)

    #Get best parameter and print it in the shell.
    bp = gs.best_params_['min_samples_leaf']
    print(f"Best parameter: {bp}")

    #Obtain the mean test scores for each parameter.
    test_cv = gs.cv_results_['mean_test_score']

    #Plot results on graph and present the best parameter on the graph as well.
    mpl.plot(parameters['min_samples_leaf'], test_cv)
    mpl.xlabel('min_samples_leaf')
    mpl.ylabel('mean roc_auc_score')
    mpl.title(f"GridSearchCV Dataset ID: {dataset_id}\nBest Parameter Value: {bp}\n")

    #Show graph.
    mpl.show()

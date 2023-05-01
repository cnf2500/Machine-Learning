#Name: Chris Forte
#Date: 09/04/2023
#Course: COMPSCI 711
#Task: Assignment II

#This program performs various functions, which can be best explained in four sections.

#The first section encompasses multiple methods dedicated to calculating four varieties of
#regression. These varieties are: linear regression, decision tree, K-nearest neighbor,
#and SVR (support vector machines).

#The second section addresses an initial task, which compares the base and bagged versions of the
#aforementioned regressions.

#The third section addresses a second task, which compares the base and boosted versions of the
#aforementioned regressions.

#The fourth section addresses a third task, which compares the base and voting ensemble versions
#of the aforementioned regressions.

#Referenced resources and tools include https://www.geeksforgeeks.org, https://www.openml.org,
#"Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples,
#and Case Studies", https://docs.python.org/, and https://chat.openai.com. 


import openml
import plotly.graph_objects as go
import pandas as pd

from plotly.subplots import make_subplots
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import ttest_rel
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate


#Data Load and Cleaning Step
def load_openml_dataset(data_id):
    dst = datasets.fetch_openml(data_id=data_id)

    ct = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), [0, 1, 6])], remainder="passthrough")

    dst_new_data = ct.fit_transform(dst.data)

    new_data = pd.DataFrame(dst_new_data, columns=ct.get_feature_names_out(), index=dst.data.index)

    return new_data, dst.target


#Base Linear Regression
def base_lr_regression():
    data, target = load_openml_dataset(43785)

    base_lr = LinearRegression()

    scores = cross_validate(base_lr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    base_lr_rmse = 0-scores["test_score"]

    return base_lr_rmse.mean()


#Boosted Linear Regression
def boosted_lr_regression():
    data, target = load_openml_dataset(43785)

    boosted_lr = AdaBoostRegressor(estimator=LinearRegression())

    scores = cross_validate(boosted_lr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    boosted_lr_rmse = 0-scores["test_score"]

    return boosted_lr_rmse.mean()


#Bagged Linear Regression
def bagged_lr_regression():
    data, target = load_openml_dataset(43785)

    bagged_lr = BaggingRegressor(estimator=LinearRegression())

    scores = cross_validate(bagged_lr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    bagged_lr_rmse = 0-scores["test_score"]

    return bagged_lr_rmse.mean()


#Ensemble Linear Regression
def ensemble_lr_regression():
    data, target = load_openml_dataset(43785)
    
    lr_vr = VotingRegressor([("lr", LinearRegression()), ("svr", SVR())])
    
    scores = cross_validate(lr_vr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    ensemble_lr_rmse = 0-scores["test_score"]

    return ensemble_lr_rmse.mean()


#Base Decision Tree Regression
def base_dt_regression():
    data, target = load_openml_dataset(43785)

    parameters = {'min_samples_leaf': [1, 3, 5, 7, 9]}

    bdtr = DecisionTreeRegressor()

    tuned_dtr = GridSearchCV(estimator=bdtr, param_grid=parameters, cv=5, scoring="neg_root_mean_squared_error")

    scores = cross_validate(tuned_dtr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    base_dt_rmse = 0-scores["test_score"]

    return base_dt_rmse.mean()


#Bagged Decision Tree Regression
def bagged_dt_regression():
    data, target = load_openml_dataset(43785)
    
    parameters = {'estimator': [DecisionTreeRegressor(min_samples_leaf=l) for l in [1, 3, 5, 7, 9]]}

    bagged_dtr = BaggingRegressor()

    tuned_dtr = GridSearchCV(estimator=bagged_dtr, param_grid=parameters, cv=5, scoring="neg_root_mean_squared_error")

    scores = cross_validate(tuned_dtr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    bagged_dt_rmse = 0-scores["test_score"]
    
    return bagged_dt_rmse.mean()


#Boosted Decision Tree Regression
def boosted_dt_regression():
    data, target = load_openml_dataset(43785)
    
    parameters = {'estimator': [DecisionTreeRegressor(min_samples_leaf=l) for l in [1, 3, 5, 7, 9]]}

    boosted_dtr = AdaBoostRegressor()
    
    tuned_dtr = GridSearchCV(estimator=boosted_dtr, param_grid=parameters, cv=5, scoring="neg_root_mean_squared_error")

    scores = cross_validate(tuned_dtr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    boosted_dt_rmse = 0-scores["test_score"]

    return boosted_dt_rmse.mean()


#Ensemble Decision Tree Regression
def ensemble_dt_regression():
    data, target = load_openml_dataset(43785)
    
    dtrs = [DecisionTreeRegressor(min_samples_leaf=l) for l in [1, 3, 5, 7, 9]]

    dt_vr = VotingRegressor([("dtr" + str(i), dtrs[i]) for i in range(len(dtrs))])

    scores = cross_validate(dt_vr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    ensemble_dt_rmse = 0-scores["test_score"]
    
    return ensemble_dt_rmse.mean()


#Base K-Nearest Neighbor Regression
def base_knn_regression():
    data, target = load_openml_dataset(43785)

    parameters = {'n_neighbors': [1, 3, 5, 7, 9]}

    base_knn = KNeighborsRegressor()

    tuned_knn = GridSearchCV(estimator=base_knn, param_grid=parameters, cv=5, scoring="neg_root_mean_squared_error")

    scores = cross_validate(tuned_knn, data, target, cv=10, scoring="neg_root_mean_squared_error")

    base_knn_rmse = 0-scores["test_score"]
    
    return base_knn_rmse.mean()


#Bagged K-Nearest Neighbor Regression
def bagged_knn_regression():
    data, target = load_openml_dataset(43785)

    parameters = {'estimator': [KNeighborsRegressor(n_neighbors=k) for k in [1, 3, 5, 7, 9]]}

    bagged_knn = BaggingRegressor()

    tuned_bknn = GridSearchCV(estimator=bagged_knn, param_grid=parameters, cv=5, scoring="neg_root_mean_squared_error")

    scores = cross_validate(tuned_bknn, data, target, cv=10, scoring="neg_root_mean_squared_error")

    bagged_knn_rmse = 0-scores["test_score"]
    
    return bagged_knn_rmse.mean()


#Boosted K-Nearest Neighbor Regression
def boosted_knn_regression():
    data, target = load_openml_dataset(43785)

    parameters = {'estimator': [KNeighborsRegressor(n_neighbors=k) for k in [1, 3, 5, 7, 9]]}

    boosted_knn = AdaBoostRegressor()
    
    tuned_knn = GridSearchCV(estimator=boosted_knn, param_grid=parameters, cv=5, scoring="neg_root_mean_squared_error")

    scores = cross_validate(tuned_knn, data, target, cv=10, scoring="neg_root_mean_squared_error")

    boosted_knn_rmse = 0-scores["test_score"]

    return boosted_knn_rmse.mean()


#Ensemble K-Nearest Neighbor Regression
def ensemble_knn_regression():
    data, target = load_openml_dataset(43785)

    parameters = [KNeighborsRegressor(n_neighbors=k) for k in [1, 3, 5, 7, 9]]

    knn_vr = VotingRegressor([("parameters" + str(i), parameters[i]) for i in range(len(parameters))])

    scores = cross_validate(knn_vr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    ensemble_knn_rmse = 0-scores["test_score"]
    
    return ensemble_knn_rmse.mean()


#Base Support-Vector Machine Regression
def base_svr_regression():
    data, target = load_openml_dataset(43785)

    svr = SVR()
    
    scores = cross_validate(svr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    base_svr_rmse = 0-scores["test_score"]

    return base_svr_rmse.mean()


#Bagged Support-Vector Machine Regression
def bagged_svr_regression():
    data, target = load_openml_dataset(43785)

    bagged_svr = BaggingRegressor(estimator=SVR())

    scores = cross_validate(bagged_svr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    bagged_svr_rmse = 0-scores["test_score"]

    return bagged_svr_rmse.mean()


#Boosted Support-Vector Machine Regression
def boosted_svr_regression():
    data, target = load_openml_dataset(43785)
    
    boosted_svr = AdaBoostRegressor(estimator=SVR())
    
    scores = cross_validate(boosted_svr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    boosted_svr_rmse = 0-scores["test_score"]

    return boosted_svr_rmse.mean()


#Ensemble Support-Vector Machine Regression
def ensemble_svr_regression():
    data, target = load_openml_dataset(43785)
    
    svr_vr = VotingRegressor([("svr", SVR())])
    
    scores = cross_validate(svr_vr, data, target, cv=10, scoring="neg_root_mean_squared_error")

    ensemble_svr_rmse = 0-scores["test_score"]

    return ensemble_svr_rmse.mean()


def taskOne():
    #Retrieve Base and Bagged RMSE values for Each Method
    lr_base_rmse = base_lr_regression()
    lr_bagged_rmse = bagged_lr_regression()
    dt_base_rmse = base_dt_regression()
    dt_bagged_rmse = bagged_dt_regression()
    knn_base_rmse = base_knn_regression()
    knn_bagged_rmse = bagged_knn_regression()
    svr_base_rmse = base_svr_regression()
    svr_bagged_rmse = bagged_svr_regression()

    #List of Method Names
    method_names = ['LR', 'DT', 'KNN', 'SVR']

    #List of Base RMSE Values
    base_rmse = [lr_base_rmse, dt_base_rmse, knn_base_rmse, svr_base_rmse]

    #List of Bagged RMSE Values
    bagged_rmse = [lr_bagged_rmse, dt_bagged_rmse, knn_bagged_rmse, svr_bagged_rmse]

    #Initialize Lists for Base and Bagged RMSE Values
    base = []
    bagged = []

    #Calculate P-Values
    lr_pval = ttest_rel([lr_base_rmse], [lr_bagged_rmse]).pvalue
    dt_pval = ttest_rel([dt_base_rmse], [dt_bagged_rmse]).pvalue
    knn_pval = ttest_rel([knn_base_rmse], [knn_bagged_rmse]).pvalue
    svr_pval = ttest_rel([svr_base_rmse], [svr_bagged_rmse]).pvalue

    #Create Table
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Table(
    header=dict(values=['Method', 'LR', 'DT', 'KNN', 'SVR']),
    cells=dict(values=[['Base', 'Bagged'],
                        [f"<b>{(lr_base_rmse):.4f}{'*' if lr_pval < 0.05 else ''}</b>" if lr_base_rmse < lr_bagged_rmse else round(lr_base_rmse, 4) if round(lr_base_rmse, 4) != round(lr_bagged_rmse, 4) else round(lr_base_rmse, 4) if lr_pval > 0.05 else f"{lr_base_rmse:.4f}*",
                         f"<b>{(lr_bagged_rmse):.4f}{'*' if lr_pval < 0.05 else ''}</b>" if lr_bagged_rmse < lr_base_rmse else round(lr_bagged_rmse, 4) if round(lr_base_rmse, 4) != round(lr_bagged_rmse, 4) else round(lr_bagged_rmse, 4) if lr_pval > 0.05 else f"{lr_bagged_rmse:.4f}*"],
                        [f"<b>{(dt_base_rmse):.4f}{'*' if dt_pval < 0.05 else ''}</b>" if dt_base_rmse < dt_bagged_rmse else round(dt_base_rmse, 4) if round(dt_base_rmse, 4) != round(dt_bagged_rmse, 4) else round(dt_base_rmse, 4) if dt_pval > 0.05 else f"{dt_base_rmse:.4f}*",
                         f"<b>{(dt_bagged_rmse):.4f}{'*' if dt_pval < 0.05 else ''}</b>" if dt_bagged_rmse < dt_base_rmse else round(dt_bagged_rmse, 4) if round(dt_base_rmse, 4) != round(dt_bagged_rmse, 4) else round(dt_bagged_rmse, 4) if dt_pval > 0.05 else f"{dt_bagged_rmse:.4f}*"],
                        [f"<b>{(knn_base_rmse):.4f}{'*' if knn_pval < 0.05 else ''}</b>" if knn_base_rmse < knn_bagged_rmse else round(knn_base_rmse, 4) if round(knn_base_rmse, 4) != round(knn_bagged_rmse, 4) else round(knn_base_rmse, 4) if knn_pval > 0.05 else f"{knn_base_rmse:.4f}*",
                         f"<b>{(knn_bagged_rmse):.4f}{'*' if knn_pval < 0.05 else ''}</b>" if knn_bagged_rmse < knn_base_rmse else round(knn_bagged_rmse, 4) if round(knn_base_rmse, 4) != round(knn_bagged_rmse, 4) else round(knn_bagged_rmse, 4) if knn_pval > 0.05 else f"{knn_bagged_rmse:.4f}*"],
                        [f"<b>{(svr_base_rmse):.4f}{'*' if svr_pval < 0.05 else ''}</b>" if svr_base_rmse < svr_bagged_rmse else round(svr_base_rmse, 4) if round(svr_base_rmse, 4) != round(svr_bagged_rmse, 4) else round(svr_base_rmse, 4) if svr_pval > 0.05 else f"{svr_base_rmse:.4f}",
                         f"<b>{(svr_bagged_rmse):.4f}{'' if svr_pval < 0.05 else ''}</b>" if svr_bagged_rmse < svr_base_rmse else round(svr_bagged_rmse, 4) if round(svr_base_rmse, 4) != round(svr_bagged_rmse, 4) else round(svr_bagged_rmse, 4) if svr_pval > 0.05 else f"{svr_bagged_rmse:.4f}*"]]
                    )
    ))

    #Layout
    fig.update_layout(title='RMSE Values for Regression Methods: Comparison of Base and Bagged Models',
                      font=dict(size=12),
                      height=400,
                      margin=dict(l=50, r=50, t=50, b=50))

    #Show Figure in Browser
    fig.show()


def taskTwo():
    #Retrieve Base and Boosted RMSE values for Each Method
    lr_base_rmse = base_lr_regression()
    lr_boosted_rmse = boosted_lr_regression()
    dt_base_rmse = base_dt_regression()
    dt_boosted_rmse = boosted_dt_regression()
    knn_base_rmse = base_knn_regression()
    knn_boosted_rmse = boosted_knn_regression()
    svr_base_rmse = base_svr_regression()
    svr_boosted_rmse = boosted_svr_regression()

    #List of Method Names
    method_names = ['LR', 'DT', 'KNN', 'SVR']

    #List of Base RMSE Values
    base_rmse = [lr_base_rmse, dt_base_rmse, knn_base_rmse, svr_base_rmse]

    #List of Boosted RMSE Values
    boosted_rmse = [lr_boosted_rmse, dt_boosted_rmse, knn_boosted_rmse, svr_boosted_rmse]

    #Initialize Lists for Base and Boosted RMSE Values
    base = []
    boosted = []

    #Calculate P-Values
    lr_pval = ttest_rel([lr_base_rmse], [lr_boosted_rmse]).pvalue
    dt_pval = ttest_rel([dt_base_rmse], [dt_boosted_rmse]).pvalue
    knn_pval = ttest_rel([knn_base_rmse], [knn_boosted_rmse]).pvalue
    svr_pval = ttest_rel([svr_base_rmse], [svr_boosted_rmse]).pvalue

    #Create Table
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Table(
    header=dict(values=['Method', 'LR', 'DT', 'KNN', 'SVR']),
    cells=dict(values=[['Base', 'Boosted'],
                        [f"<b>{(lr_base_rmse):.4f}{'*' if lr_pval < 0.05 else ''}</b>" if lr_base_rmse < lr_boosted_rmse else round(lr_base_rmse, 4) if round(lr_base_rmse, 4) != round(lr_boosted_rmse, 4) else round(lr_base_rmse, 4) if lr_pval > 0.05 else f"{lr_base_rmse:.4f}*",
                         f"<b>{(lr_boosted_rmse):.4f}{'*' if lr_pval < 0.05 else ''}</b>" if lr_boosted_rmse < lr_base_rmse else round(lr_boosted_rmse, 4) if round(lr_base_rmse, 4) != round(lr_boosted_rmse, 4) else round(lr_boosted_rmse, 4) if lr_pval > 0.05 else f"{lr_boosted_rmse:.4f}*"],
                        [f"<b>{(dt_base_rmse):.4f}{'*' if dt_pval < 0.05 else ''}</b>" if dt_base_rmse < dt_boosted_rmse else round(dt_base_rmse, 4) if round(dt_base_rmse, 4) != round(dt_boosted_rmse, 4) else round(dt_base_rmse, 4) if dt_pval > 0.05 else f"{dt_base_rmse:.4f}*",
                         f"<b>{(dt_boosted_rmse):.4f}{'*' if dt_pval < 0.05 else ''}</b>" if dt_boosted_rmse < dt_base_rmse else round(dt_boosted_rmse, 4) if round(dt_base_rmse, 4) != round(dt_boosted_rmse, 4) else round(dt_boosted_rmse, 4) if dt_pval > 0.05 else f"{dt_boosted_rmse:.4f}*"],
                        [f"<b>{(knn_base_rmse):.4f}{'*' if knn_pval < 0.05 else ''}</b>" if knn_base_rmse < knn_boosted_rmse else round(knn_base_rmse, 4) if round(knn_base_rmse, 4) != round(knn_boosted_rmse, 4) else round(knn_base_rmse, 4) if knn_pval > 0.05 else f"{knn_base_rmse:.4f}*",
                         f"<b>{(knn_boosted_rmse):.4f}{'*' if knn_pval < 0.05 else ''}</b>" if knn_boosted_rmse < knn_base_rmse else round(knn_boosted_rmse, 4) if round(knn_base_rmse, 4) != round(knn_boosted_rmse, 4) else round(knn_boosted_rmse, 4) if knn_pval > 0.05 else f"{knn_boosted_rmse:.4f}*"],
                        [f"<b>{(svr_base_rmse):.4f}{'*' if svr_pval < 0.05 else ''}</b>" if svr_base_rmse < svr_boosted_rmse else round(svr_base_rmse, 4) if round(svr_base_rmse, 4) != round(svr_boosted_rmse, 4) else round(svr_base_rmse, 4) if svr_pval > 0.05 else f"{svr_base_rmse:.4f}",
                         f"<b>{(svr_boosted_rmse):.4f}{'' if svr_pval < 0.05 else ''}</b>" if svr_boosted_rmse < svr_base_rmse else round(svr_boosted_rmse, 4) if round(svr_base_rmse, 4) != round(svr_boosted_rmse, 4) else round(svr_boosted_rmse, 4) if svr_pval > 0.05 else f"{svr_boosted_rmse:.4f}*"]]
                    )
    ))

    #Layout
    fig.update_layout(title='RMSE Values for Regression Methods: Comparison of Base and Boosted Models',
                      font=dict(size=12),
                      height=400,
                      margin=dict(l=50, r=50, t=50, b=50))

    #Show Figure in Browser
    fig.show()


def taskThree():
    #Retrieve Base and Voting Ensemble RMSE values for Each Method
    lr_base_rmse = base_lr_regression()
    lr_ensemble_rmse = ensemble_lr_regression()
    dt_base_rmse = base_dt_regression()
    dt_ensemble_rmse = ensemble_dt_regression()
    knn_base_rmse = base_knn_regression()
    knn_ensemble_rmse = ensemble_knn_regression()
    svr_base_rmse = base_svr_regression()
    svr_ensemble_rmse = ensemble_svr_regression()

    #List of Method Names
    method_names = ['LR', 'DT', 'KNN', 'SVR']

    #List of Base RMSE Values
    base_rmse = [lr_base_rmse, dt_base_rmse, knn_base_rmse, svr_base_rmse]

    #List of Voting Ensemble RMSE Values
    ensemble_rmse = [lr_ensemble_rmse, dt_ensemble_rmse, knn_ensemble_rmse, svr_ensemble_rmse]

    #Initialize Lists for Base and Voting Ensemble RMSE Values
    base = []
    ensemble = []

    #Calculate P-Values
    lr_pval = ttest_rel([lr_base_rmse], [lr_ensemble_rmse]).pvalue
    dt_pval = ttest_rel([dt_base_rmse], [dt_ensemble_rmse]).pvalue
    knn_pval = ttest_rel([knn_base_rmse], [knn_ensemble_rmse]).pvalue
    svr_pval = ttest_rel([svr_base_rmse], [svr_ensemble_rmse]).pvalue

    #Create Table
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Table(
    header=dict(values=['Method', 'LR', 'DT', 'KNN', 'SVR']),
    cells=dict(values=[['Base', 'Voted Ensemble'],
                        [f"<b>{(lr_base_rmse):.4f}{'*' if lr_pval < 0.05 else ''}</b>" if lr_base_rmse < lr_ensemble_rmse else round(lr_base_rmse, 4) if round(lr_base_rmse, 4) != round(lr_ensemble_rmse, 4) else round(lr_base_rmse, 4) if lr_pval > 0.05 else f"{lr_base_rmse:.4f}*",
                         f"<b>{(lr_ensemble_rmse):.4f}{'*' if lr_pval < 0.05 else ''}</b>" if lr_ensemble_rmse < lr_base_rmse else round(lr_ensemble_rmse, 4) if round(lr_base_rmse, 4) != round(lr_ensemble_rmse, 4) else round(lr_ensemble_rmse, 4) if lr_pval > 0.05 else f"{lr_ensemble_rmse:.4f}*"],
                        [f"<b>{(dt_base_rmse):.4f}{'*' if dt_pval < 0.05 else ''}</b>" if dt_base_rmse < dt_ensemble_rmse else round(dt_base_rmse, 4) if round(dt_base_rmse, 4) != round(dt_ensemble_rmse, 4) else round(dt_base_rmse, 4) if dt_pval > 0.05 else f"{dt_base_rmse:.4f}*",
                         f"<b>{(dt_ensemble_rmse):.4f}{'*' if dt_pval < 0.05 else ''}</b>" if dt_ensemble_rmse < dt_base_rmse else round(dt_ensemble_rmse, 4) if round(dt_base_rmse, 4) != round(dt_ensemble_rmse, 4) else round(dt_ensemble_rmse, 4) if dt_pval > 0.05 else f"{dt_ensemble_rmse:.4f}*"],
                        [f"<b>{(knn_base_rmse):.4f}{'*' if knn_pval < 0.05 else ''}</b>" if knn_base_rmse < knn_ensemble_rmse else round(knn_base_rmse, 4) if round(knn_base_rmse, 4) != round(knn_ensemble_rmse, 4) else round(knn_base_rmse, 4) if knn_pval > 0.05 else f"{knn_base_rmse:.4f}*",
                         f"<b>{(knn_ensemble_rmse):.4f}{'*' if knn_pval < 0.05 else ''}</b>" if knn_ensemble_rmse < knn_base_rmse else round(knn_ensemble_rmse, 4) if round(knn_base_rmse, 4) != round(knn_ensemble_rmse, 4) else round(knn_ensemble_rmse, 4) if knn_pval > 0.05 else f"{knn_ensemble_rmse:.4f}*"],
                        [f"<b>{(svr_base_rmse):.4f}{'*' if svr_pval < 0.05 else ''}</b>" if svr_base_rmse < svr_ensemble_rmse else round(svr_base_rmse, 4) if round(svr_base_rmse, 4) != round(svr_ensemble_rmse, 4) else round(svr_base_rmse, 4) if svr_pval > 0.05 else f"{svr_base_rmse:.4f}",
                         f"<b>{(svr_ensemble_rmse):.4f}{'' if svr_pval < 0.05 else ''}</b>" if svr_ensemble_rmse < svr_base_rmse else round(svr_ensemble_rmse, 4) if round(svr_base_rmse, 4) != round(svr_ensemble_rmse, 4) else round(svr_ensemble_rmse, 4) if svr_pval > 0.05 else f"{svr_ensemble_rmse:.4f}*"]]
                    )
    ))

    #Layout
    fig.update_layout(title='RMSE Values for Regression Methods: Comparison of Base and Voted Ensemble Models',
                      font=dict(size=12),
                      height=400,
                      margin=dict(l=50, r=50, t=50, b=50))

    #Show Figure in Browser
    fig.show()

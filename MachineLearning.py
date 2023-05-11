from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

import re
import numpy as np
import pandas as pd


def rdforest(X_train, X_test, y_train, y_test):
    '''
    Creates multiple Random Forest classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.

    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    df = pd.DataFrame(
        columns=['N_Estimators', 'Max_Depth', 'Confusion Matrix'])

    estimator = 400
    max_depth = 7

    rdf = RandomForestClassifier(
        n_estimators=estimator, max_depth=max_depth, random_state=0, n_jobs=-1)
    rdf.fit(X_train, y_train)
    predicted_scores = rdf.predict_proba(X_test)[:, 1]
    predicted_labels = (predicted_scores > 0.5).astype('int32')
    # tn, fp, fn, tp = confusion_matrix(
    #     y_test, predicted_labels, labels=[0, 1]).ravel()
    # convert_matrix = [tn, fp, fn, tp]
    # rows.append([estimator, max_d, convert_matrix])
    print("Prediction:")
    unique, counts = np.unique(predicted_labels, return_counts=True)
    print(dict(zip(unique, counts)))
    print(classification_report(y_test, predicted_labels))

    fpr, tpr, _ = roc_curve(y_test, predicted_scores)

    # for i in range(len(rows)):
    #     df = df.append({'N_Estimators': rows[i][0], 'Max_Depth': rows[i]
    #                    [1], 'Confusion Matrix': rows[i][2]}, ignore_index=True)
    # return df
    return fpr, tpr, auc(fpr, tpr)


def xgboost(X_train, X_test, y_train, y_test):
    '''
    Creates multiple XgBoost classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.

    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    df = pd.DataFrame(
        columns=['Max_depth', 'N_estimators', 'Confusion Matrix'])
    rows = []
    rate = 0.05
    depth = 5
    estimators = 300

    xgb = XGBClassifier(booster='gbtree', max_depth=depth, learning_rate=rate,
                        n_estimators=estimators)
    xgb.fit(X_train, y_train)
    predicted_scores = xgb.predict_proba(X_test)[:, 1]
    predicted_labels = (predicted_scores > 0.5).astype('int32')
    # tn, fp, fn, tp = confusion_matrix(
    #     y_test, predicted_labels, labels=[0, 1]).ravel()
    # convert_matrix = [tn, fp, fn, tp]
    # rows.append([depth, estimators, convert_matrix])
    print("Prediction:")
    unique, counts = np.unique(predicted_labels, return_counts=True)
    print(dict(zip(unique, counts)))
    print(classification_report(y_test, predicted_labels))

    fpr, tpr, _ = roc_curve(y_test, predicted_scores)
    # for i in range(len(rows)):
    #     df = df.append({'Max_depth': rows[i][0], 'N_estimators': rows[i][1],
    #                     'Confusion Matrix': rows[i][2]}, ignore_index=True)
    # return df
    return fpr, tpr, auc(fpr, tpr)


def naive_bayes(X_train, X_test, y_train, y_test):
    '''
    Creates multiple Naive Bayes classifier models on X_train and y_train with different parameters.
    Runs the models on X_test and compare the results with y_test.

    Args:
        X_train: A Numpy array containing the dataset for training
        X_test: A Numpy array containing the dataset for testing
        y_train: A Numpy array consisting of the target values for training
        y_test: A Numpy array consisting of the target values for testing

    Returns:
        A DataFrame with all the paramaters used and confusion matrices of each model
    '''
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    predicted_scores = bnb.predict_proba(X_test)[:, 1]
    predicted_labels = (predicted_scores > 0.5).astype('int32')
    print("Prediction:")
    unique, counts = np.unique(predicted_labels, return_counts=True)
    print(dict(zip(unique, counts)))
    print(classification_report(y_test, predicted_labels))

    fpr, tpr, _ = roc_curve(y_test, predicted_scores)
    return fpr, tpr, auc(fpr, tpr)
    # tn, fp, fn, tp = confusion_matrix(
    #     y_test, predicted_labels, labels=[0, 1]).ravel()
    # convert_matrix_b = [tn, fp, fn, tp]

    # df = df.append({'Confusion Matrix': convert_matrix_b}, ignore_index=True)
    # return df

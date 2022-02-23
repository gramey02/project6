"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler


def test_updates():
    """
    Check training of the model
    """
    #create a logistic regression object with the following data
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 
                                                                 'Computed tomography of chest and abdomen',
                                                                 'Plain chest X-ray (procedure)',
                                                                 'Low Density Lipoprotein Cholesterol',
                                                                 'Creatinine', 
                                                                 'AGE_DIAGNOSIS'], 
                                                       split_percent=0.8, split_state=42)
    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform (X_val)
    
    #create object
    log_model = logreg.LogisticRegression(num_feats=6, max_iter=250, tol=0.0001, learning_rate=0.05, batch_size=12)
    #train model
    log_model.train_model(X_train, y_train, X_val, y_val)
    
    #Check that your gradient is being calculated correctly (check if the gradient is always negative, and if it is decreasing?)
    
    
    #Check that your loss function is correct and that you have reasonable losses at the end of training 
    #(define reasonable loss as under 150)
    
    #Check to see if your losses approach zero (look at the loss_history_train vector)
    
    

    pass

def test_predict():
    """
    Check testing of the model
    """
    #Check that the weights are being updated (check if original weights before training and new weights after training are equal)
    
    #Check that reasonable estimates are given for NSCLC classification (i.e. all values are between 0 and 1)

    # Check accuracy of model after training
    #(define an accuracy function which calculates the number of correct classifications and divides it by the # of total outcomes)

    pass
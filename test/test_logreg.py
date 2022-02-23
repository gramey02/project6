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
    log_model = logreg.LogisticRegression(num_feats=6, max_iter=150, tol=0.0001, learning_rate=0.05, batch_size=12)
    #train model
    log_model.train_model(X_train, y_train, X_val, y_val)
    
    
    
    
    #Check that your gradient is being calculated correctly
    gradients = log_model.grad_history #get the stored gradient steps
    #assert that the first array of gradients is not the same as the last array of gradients (i.e. assert that the gradients changed)
    comparison = gradients[0]==gradients[len(gradients)-1]
    arrays_are_equal = comparison.all() #compares all array values to make sure they are equal at the same indices
    assert arrays_are_equal==False
    
    
    
    
    #Check that your loss function is correct and that you have reasonable losses at the end of training
    #i.e. check that the min training and validation loss are under 3 and 200, respectively with the given hyperparameters
    assert min(log_model.loss_history_train)<5
    assert min(log_model.loss_history_val)<200
    
    
    
    
    #Check to see if your training losses approach zero (look at the loss_history_train vector) and are generally decreasing
    prev_num = 0.0
    current_num = 0.0
    for i in range(0,len(log_model.loss_history_train)):
        current_num = log_model.loss_history_train[i]
        #tends to stabilize around i=200, so start checking for decreasing values there
        if i>200:
            assert current_num<prev_num
        prev_num = current_num
        
        

def test_predict():
    """
    Check testing of the model
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
    log_model = logreg.LogisticRegression(num_feats=6, max_iter=150, tol=0.0001, learning_rate=0.05, batch_size=12)
    original_weights = log_model.W #capture initialized weights
    #train model
    log_model.train_model(X_train, y_train, X_val, y_val)
    new_weights = log_model.W #capture final weights
    
    
    
    
    #Check that the weights are being updated (i.e. check if original weights before training and new weights are NOT equal)
    comparison = original_weights==new_weights
    arrays_are_equal = comparison.all() #compares all array values to make sure they are equal at the same indices
    assert arrays_are_equal==False
    
    
    
    
    #Check that reasonable estimates are given for NSCLC classification (i.e. all values are between 0 and 1)
    y_pred = log_model.make_prediction(X_train)
    assert np.amin(y_pred)>=0
    assert np.amax(y_pred)<=1
    
    
    
    
    # Check accuracy of model after training
    y_classified = y_pred
    #use cutoff value of 0.5 for binary classification
    for i in range(0,len(y_pred)):
        if y_pred[i]<0.5:
            y_classified[i] = 0
        else:
            y_classified[i] = 1

    def accuracy():
        
    #(define an accuracy function which calculates the number of correct classifications and divides it by the # of total outcomes)

    pass
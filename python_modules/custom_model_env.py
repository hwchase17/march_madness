#!/usr/bin/env python

""" custom_model_env.py: 
This is a module that defines a custom environment for working with different models.
The idea is that different models can be tweaked in this script to fit different use cases and then
run directly from this module in, say, a Jupyter Notebook.
"""

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from functions import logloss, spread_to_percent
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np

class custom_model_env:
    
    def __init__(self, X, Y, O_function):
        """
        input:
            X: training data (pandas dataframe, each row a separate data point)
            Y: training labels (np.array of match outcomes)
            O_function: maps response variables to outcomes used for optimization of hyperparameters.
        """
        
        # Save data and save outcomes as 'O'
        self.X = X
        self.Y = Y
        self.O = np.array([O_function(y) for y in Y])
        self.eva = []
        
        # Split into training and cv data
        self.X_train = self.X
        self.Y_train = self.Y
        self.O_train = self.O
        self.X_cv = None
        self.Y_cv = None
        self.O_cv = None
    
    def update_cv(self, cv):
        """
        input:
            cv: indices to be removed form self.X and self.Y
        """
        
        # Delete 'cv' indices from the train indices
        train_indices = list(self.X_train.index)
        test_indices = sorted(cv, reverse=True)
        for index in test_indices:
            del train_indices[index]
            
        # Update data structures
        self.X_train = self.X.loc[train_indices,]
        self.Y_train = self.Y[train_indices]
        self.O_train = self.O[train_indices]
        self.X_cv = self.X.loc[test_indices,]
        self.Y_cv = self.Y[test_indices]
        self.O_cv = self.O[test_indices]
    
    def get_splits(self, n_splits, cv=None):
        """
        Gets splits according to some cv data.
        
        input:
            cv: indices from X to use as cross-validation data
            
        output:
            splits to use for cross-validation (list of tuples of (train_indices, test_indices))  
        """
        
        # If 'cv' is given, define splits according to 'cv'; else use stratification
        if cv:
            train_indices = list(self.X_train.index)
            test_indices = sorted(cv, reverse=True)
            for index in test_indices:
                del train_indices[index]
            splits = [(train_indices, test_indices)]
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            splits = list(skf.split(self.X_train, self.O_train))
        
        return splits
    
    def run_xgb_grid_search(self, params, loss_function, n_splits, num_boost_round, early_stopping=None, cv=None):
        """
        Performs a grid search on a list of xgb parameters.
        
        Usual grid search algorithms optimize hyperparameters on the response variables of the model;
        this module seeks to optimize rather on a predetermined list of outcomes, with a function
        mapping response variables to those outcomes.
        
        input:
            params:           list of tuples of the form (parameter name, ['values'])
            loss_function:    first argument: list of outcomes, second argument: list of predictions
            n_splits:         number of splits to be done for stratified k-folding
            num_boost_round:  maximum number of boosting rounds
            early_stopping:   stops boosting if no improvement for 'early_stopping' rounds
            cv:               indices from X to use as cross-validation data
            
        If no cross-validation indices are specified, the cross-validation will be done using sklearn's 'StratifiedKFold'.
        If cross-validation indices are specified, 'n_splits' must be set to the number of cross-validation folds in 'cv'.
        If the instance has already been updated with cross-validation data using 'update_cv', those indices will be included
        only through evaluation and only if they are specified in the 'cv' variable.   
        
        output:
            best_param:            the optimal parameters (dictionary)
            best_num_boost_round:  the optimal number of boosting rounds
            min_loss_value:        the loss value achieved with best_param and best_num_boost_round
            
        Note: A model is saved for every split. The best parameters are defined as those that minimize the average loss function
        across the models (with each model being run on its respective 'cv' split).
        
        Currently set up to give loss as the error on match winner prediction (error defined by a loss function),
        but training fits to actual point spreads so as to not lose information during training.
        """
    
        splits = self.get_splits(n_splits, cv)
    
        # Convert data to 'DMatrix' type
        X_ = xgb.DMatrix(self.X_train, label=self.Y_train)

        # Find all parameter combinations and format list of parameters
        param_combs = list(itertools.product(*[x[1] for x in params]))
        param_dict = {}
        param_list = [x[0] for x in params]
        no_combs = len(param_combs)

        # Save a list of loss values for each parameter combination
        loss_values = [[] for _ in xrange(no_combs)]
        
        # Initialize a minimum loss value variable
        min_loss_value = float('inf')
        
        print "Fitting " + str(no_combs) + " parameter combinations."
        
        for i, param_comb in enumerate(param_combs):
        
            # Set parameters for current combination
            for j, param in enumerate(param_list):
                param_dict[param] = param_comb[j]

            # Initialize variables before boosting a given combination
            boost_rounds_left = num_boost_round
            models = [None] * n_splits
            increase_no = 0
            last_test_loss = float('inf')
            
            # Start boosting
            while boost_rounds_left > 0:

                # Define indices for splits and initialize loss and train variables to be averaged later
                splits_index = 0
                losses_train = 0
                losses_test = 0
                
                for train_index, test_index in splits:

                    train_index_real = list(self.X_train.index[train_index])
                    test_index_real = list(self.X_train.index[test_index])
                    
                    X_train = X_.slice(train_index_real)
                    Y_train = self.Y_train[train_index]
                    O_train = self.O_train[train_index]
                    X_cv = X_.slice(test_index_real)
                    Y_cv = self.Y_train[test_index]
                    O_cv = self.O_train[test_index]

                    models[splits_index] = xgb.train(param_dict, X_train, num_boost_round=1, xgb_model=models[splits_index])

                    # Get the errors from the models on the respective splits
                    preds_train = spread_to_percent(models[splits_index].predict(X_train))
                    preds_test = spread_to_percent(models[splits_index].predict(X_cv))
                    losses_train += loss_function(O_train, preds_train)
                    losses_test += loss_function(O_cv, preds_test)
                    
                    splits_index += 1
                
                loss_value_train = losses_train / n_splits
                loss_value_test = losses_test / n_splits
                
                loss_values[i].append( (loss_value_train, loss_value_test) )
                
                boost_rounds_left -= 1
                
                # If there is a new lower loss value, update the best parameters to current parameters
                if loss_value_test < min_loss_value:
                    min_loss_value = loss_value_test
                    best_param = param_dict.copy()
                    best_num_boost_round = num_boost_round - boost_rounds_left
            
                # Check for an increase in loss value
                if loss_value_test <= last_test_loss:
                    increase_no = 0
                else:
                    increase_no += 1
                
                # Break from the current boosting process if loss value has not decreased in the last 'early_stopping' rounds
                if early_stopping:
                    if increase_no == early_stopping:
                        break
            
                last_test_loss = loss_value_test
            
            print "Combination " + str(i + 1) + " completed."
            
        # Update the evaluation errors of the instance
        self.eva = pd.DataFrame(loss_values)
        
        return best_param, best_num_boost_round, min_loss_value
    
    def logreg(self, n_splits, loss_function, cv=None):
        """
        Performs a logistic regression. 
        
        input:
            loss_function:    first argument: list of outcomes, second argument: list of predictions
            n_splits:         number of splits to be done for stratified k-folding
            cv:               indices from X to use as cross-validation data
        
        output:
            loss_value_train: loss value for train set
            loss_value_test:  loss value for test set
            loss_value_cv:    loss value for cv set
            
        Currently set up to give loss as the error on match winner prediction (error defined by a loss function),
        but training fits to actual point spreads so as to not lose information during training.
        """
        splits = self.get_splits(n_splits, cv)
        
        model = LinearRegression()
        
        models = [None] * n_splits
        
        # Define indices for splits and initialize loss and train variables to be averaged later
        splits_index = 0
        losses_train = 0
        losses_test = 0
        
        for train_index, test_index in splits:

            train_index_real = list(self.X_train.index[train_index])
            test_index_real = list(self.X_train.index[test_index])
            
            X_train = self.X_train.loc[train_index_real,:]
            Y_train = self.Y_train[train_index]
            O_train = self.O_train[train_index]
            X_cv = self.X_train.loc[test_index_real,:]
            Y_cv = self.Y_train[test_index]
            O_cv = self.O_train[test_index]

            models[splits_index] = model.fit(X_train, Y_train)

            # Get the errors from the models on the respective splits
            preds_train = spread_to_percent(models[splits_index].predict(X_train))
            preds_test = spread_to_percent(models[splits_index].predict(X_cv))
            losses_train += loss_function(O_train, preds_train)
            losses_test += loss_function(O_cv, preds_test)

            splits_index += 1

        loss_value_train = losses_train / n_splits
        loss_value_test = losses_test / n_splits
        
        # Get the error for the cv data
        model_for_cv = model.fit(self.X_train, self.Y_train)
        preds_cv = spread_to_percent(model_for_cv.predict(self.X_cv))
        loss_value_cv = loss_function(self.O_cv, preds_cv)
        
        return self.X_train, self.X_cv, self.Y_train, self.O_cv, loss_value_train, loss_value_test, loss_value_cv

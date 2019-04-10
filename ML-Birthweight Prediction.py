# -*- coding: utf-8 -*-
"""
Created on March 03, 2019 Sunday
Creator : Team_11
University: Hult International Business School, San Francisco, CA 
Degree: Masters of Business Analytics, Dual Degree
Course: Machine Learning
Professor : Chase Kusterer
Module: B

Purpose: Group Project Submission by finding R^2 and the maximum predictive 
score
"""



#Importing Required Packages
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
from sklearn.model_selection import train_test_split # train/test split
import statsmodels.formula.api as sm # regression modeling

#Importing file to the work terminal
file = 'birthweight_feature_set.xlsx'

#made the work path file name to work more easily
birthweight = pd.read_excel(file)

###############################################################################
# Univariate Regression Analysis
###############################################################################

# Building a Regression Base

bwght_mage = sm.ols(formula = 
                    """bwght ~ birthweight['mage']+
                     birthweight['fage']+birthweight['meduc']+
                     birthweight['monpre']+birthweight['npvis']+
                     birthweight['cigs']+birthweight['drink']+
                     birthweight['feduc']+birthweight['fmaps']+
                     birthweight['mwhte']""",
                         data = birthweight)


# Fitting Results
results = bwght_mage.fit()

# Printing Summary Statistics
print(results.summary())




###############################################################################
# Best Predictive Model for team: Linear Regression
###############################################################################

X   = birthweight.drop([
        'bwght', 'monpre', 'omaps', 'fmaps', 'male', 
                        'mwhte', 'mblck', 'moth'], 
                        axis = 1)


y = birthweight.loc[:, 'bwght']


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.10,
                                                    random_state=508)


lg_mod = LinearRegression(fit_intercept = False)


# Fitting the model
lr_mod_fit = lr_mod.fit(X_train, y_train)

# Predictions
lr_mod_pred = lr_mod_fit.predict(X_test)

#scoring the model
lr_mod_yscore = lr_mod_fit.score(X_test, y_test)
print(lr_mod_yscore)
#output:0.6198159480811648
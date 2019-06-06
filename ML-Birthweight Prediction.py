
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 03, 2019 Sunday
Creator : Team Project
        Dang Nhat Le
        Ishwor Bhusal
        Sneha
        Cesar Mauricio
        Melis
        Ankur
University: Hult International Business School, San Francisco, CA 
Degree: Masters of Business Analytics, Dual Degree
Course: Machine Learning
Professor : Chase Kusterer
Module: B

Purpose: Group Project Submission by finding the maximum predictive score
"""

#Importing Required Libraries and Packages

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as sm # regression modeling
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.model_selection import cross_val_score # k-folds cross validation


#Importing file to the work terminal
file = 'birthweight_feature_set.xlsx'

#made the work path file name to work more easily
birthweight = pd.read_excel(file)



#############################################################################
##############################################################################
# Working on Dataset Exploration
##############################################################################
##############################################################################


# Listing the name of the columns 
birthweight.columns

# Displaying the first rows of the DataFrame
print(birthweight.head())

# Dimensions of the DataFrame
birthweight.shape

# Information about each variable
birthweight.info()

# Descriptive statistics
birthweight.describe().round(2)

#Shorting the values according the birthweight higher to lower
birthweight.sort_values('bwght', ascending = False)



###############################################################################
# Imputing Missing Values
###############################################################################

#First checking the missing the values in the dataset
print(
      birthweight
      .isnull()
      .sum()
      )



for col in birthweight:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if birthweight[col].isnull().any():
        birthweight['m_'+col] = birthweight[col].isnull().astype(int)



##############################################################################
# Missing values being filled with the median
##############################################################################

fill = birthweight['fage'].median()

birthweight['fage'] = birthweight['fage'].fillna(fill)

fill = birthweight['omaps'].median()

birthweight['omaps'] = birthweight['omaps'].fillna(fill)


##############################################################################
# Checking the overall dataset to see if there are any remaining missing values
##############################################################################

print(
      birthweight
      .isnull()
      .any()
      .any()
      )

#There is no missing values anymore


###############################################################################
# Quantiles Analytis:
###############################################################################

birthweight_quantiles = birthweight.loc[:, :].quantile([0.20,
                                                0.40,
                                                0.60,
                                                0.80,
                                                1.00])

    
print(birthweight_quantiles)

for col in birthweight:
    print(col)

##############################################################################
# Visual EDA (Histograms) in order to visualize the dataset and insights
##############################################################################

plt.subplot(2, 2, 1)
sns.distplot(birthweight['mage'],
             bins = 10,
             kde = True,
             rug = True,
             color = 'g')

plt.xlabel('Mothers Age')
########################
plt.subplot(2, 2, 2)
sns.distplot(birthweight['meduc'],
             bins = 10,
             kde = True,
             rug = True,
             color = 'y')

plt.xlabel('Mothers Education')
########################
plt.subplot(2, 2, 3)
sns.distplot(birthweight['npvis'],
             bins = 10,
             kde = True,
             rug = True,
             color = 'orange')

plt.xlabel('npvis')
#######################
plt.subplot(2, 2, 4)

sns.distplot(birthweight['fage'],
             bins = 10,
             kde = True,
             rug = True,
             color = 'r')

plt.xlabel('Fathers Age')

plt.tight_layout()
plt.savefig('Birthweight Data Histograms.png')

plt.show()


##############################################################################


plt.subplot(2, 2, 1)
sns.distplot(birthweight['cigs'],
             bins = 10,
             kde = True,
             color = 'g')

plt.xlabel('Mother Smoking Cigarette')
########################
plt.subplot(2, 2, 2)
sns.distplot(birthweight['drink'],
             bins = 10,
             kde = True,
             color = 'y')

plt.xlabel('Mother Drinking Alcohol')

plt.tight_layout()
plt.savefig('Birthweight Data Histograms 1 of 2.png')

plt.show()



###############################################################################
# Tuning and Flagging Outliers
###############################################################################

birthweight_quantiles = birthweight.loc[:, :].quantile([0.05,
                                                0.40,
                                                0.60,
                                                0.80,
                                                0.95])


#Flagging outliers for Mother's Age

mage_lo = 18
mage_hi = 50   

birthweight['out_mage'] = 0

for val in enumerate(birthweight.loc[:,'mage']):
    if val[1]<= mage_lo: birthweight.loc[val[0],'out_mage'] = -1
    
for val in enumerate(birthweight.loc[:,'mage']):
    if val[1]>= mage_hi: birthweight.loc[val[0],'out_mage'] = 1
    
##############################################################################


#Flagging outliers for Number of Prenatal Visits:

npvis_lo = 5
npvis_hi = 30  

birthweight['out_npvis'] = 0

for val in enumerate(birthweight.loc[:,'npvis']):
    if val[1]<= npvis_lo: birthweight.loc[val[0],'out_npvis'] = -1
    
for val in enumerate(birthweight.loc[:,'npvis']):
    if val[1]>= npvis_hi: birthweight.loc[val[0],'out_npvis'] = 1


###############################################################################


#Flagging outliers for Number of Prenatal Visits:
npvis_lo = 5
npvis_hi = 30  

birthweight['out_npvis'] = 0

for val in enumerate(birthweight.loc[:,'npvis']):
    if val[1]<= npvis_lo: birthweight.loc[val[0],'out_npvis'] = -1
    
for val in enumerate(birthweight.loc[:,'npvis']):
    if val[1]>= npvis_hi: birthweight.loc[val[0],'out_npvis'] = 1



###############################################################################
# Correlation Analysis
###############################################################################


birthweight.head()

df_corr = birthweight.corr().round(2)

print(df_corr)

df_corr.loc['bwght'].sort_values(ascending = False)

#cigs,m drink, mage and fage are the best variables for the prediction
"""
bwght      1.00
omaps      0.25
fmaps      0.25
feduc      0.13
mblck      0.13
fblck      0.12
male       0.11
meduc      0.09
m_npvis    0.06
npvis      0.06
m_feduc   -0.00
moth      -0.02
fwhte     -0.04
monpre    -0.05
foth      -0.08
mwhte     -0.11
m_meduc   -0.13
fage      -0.40
mage      -0.46
cigs      -0.57
drink     -0.74
"""


##############################################################################
# Correlation Heatmap
##############################################################################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('Birthweight Correlation Heatmap.png')
plt.show()

#############################################################################
#############################################################################
#                 Now time to Build up the prediction models
#############################################################################
#############################################################################



##############################################################################
#                               Base Model
##############################################################################

X   = birthweight.drop(['bwght'], axis = 1)

y = birthweight.loc[:, 'bwght']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.10,
                                                    random_state=508)



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
#Running KNeightbor Regressor
###############################################################################

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X   = birthweight.drop(['bwght', 'feduc', 'omaps', 'fmaps',
                        'male', 'foth', 'out_mage', 'out_npvis'], axis = 1)

y = birthweight.loc[:, 'bwght']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.10,
                                                    random_state=508)

# Instantiating a model with k = 5
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 14)


# Fitting the model based on the training data
knn_reg.fit(X_train, y_train)

# Scoring the model
y_score = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)
#output: 0.4900115023210361




##############################################################################
#Instantiating a KNN regressor object
##############################################################################

#Model

X   = birthweight.drop(['bwght', 'fage', 'fmaps', 
                        'mblck', 'moth', 'fwhte', 'fblck', 'foth'], axis = 1)

y = birthweight.loc[:, 'bwght']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.10,
                                                    random_state=508)


knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 5)


# Checking the type of this new object
type(knn_reg)

#Teaching (fitting) the algorithm based on the training data
knn_reg.fit(X_train, y_train)

#Predicting on the X_data that the model has never seen before
y_pred = knn_reg.predict(X_test)
    
# Printing out prediction values for each test observation
print(f"""
Test set predictions:
{y_pred[0:18]}
""")
    
    
#Calling the score method, which compares the predicted values to the actual values.
y_score = knn_reg.score(X_test, y_test)


# The score is directly comparable to R-Square
print(y_score)
#Output: 0.46591928466083826




###############################################################################
#Using KNN on the Optimal Variables
##############################################################################

# Exact loop as before
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

##############################################################################
print("The optimal number of neighbors is", \
      test_accuracy.index(max(test_accuracy)), \
      "with an optimal score of", \
      max(test_accuracy))
#Output: The optimal number of neighbors is 4 with an optimal score of 0.4879509770000522




###############################################################################
#Building a KNN model based on above Model
###############################################################################

X   = birthweight.drop([
        'bwght', 'fage', 'feduc', 'omaps', 'fmaps',
                        'male', 'mwhte', 'mblck', 'moth'], axis = 1)

y = birthweight.loc[:, 'bwght']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.10,
                                                    random_state=508)

# Building a model with k = 5
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 5)


# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)

# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_knn_optimal)

# Generating Predictions based on the optimal KNN model
knn_reg_optimal_pred = knn_reg_fit.predict(X_test)
#Output: 0.43301453370935306




##############################################################################
#Predicting Now on the OLS Regression
##############################################################################

#Fixing the Data (X) and Target (y)

X   = birthweight.drop(['bwght'
                        ], axis = 1)

y = birthweight.loc[:, 'bwght']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.10,
                                                    random_state=508)
    
from sklearn.linear_model import LinearRegression
# Prepping the Model
lr = LinearRegression(fit_intercept = False)


# Fitting the model
lr_fit = lr.fit(X_train, y_train)


# Predictions
lr_pred = lr_fit.predict(X_test)

# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score_ols_optimal)
#0.5958968692209615




###############################################################################
# Prepping the Model Linear Regression
###############################################################################

X   = birthweight.drop(
        ['bwght', 'monpre', 'omaps', 'fmaps', 'male', 
                        'mwhte', 'mblck', 'moth'], 
         axis = 1)


y = birthweight.loc[:, 'bwght']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.10,
                                                    random_state=508)


lr_mod = LinearRegression(fit_intercept = False)


# Fitting the model
lr_mod_fit = lr_mod.fit(X_train, y_train)


# Predictions
lr_mod_pred = lr_mod_fit.predict(X_test)


#scoring the model
lr_mod_yscore = lr_mod_fit.score(X_test, y_test)
print(lr_mod_yscore)
#output:0.6198159480811648

##############################################################################
##############################################################################


#Now we will export the predictions to excel sheet for submission

team_11_model_prediction = pd.DataFrame({'Actual' : y_test,
                                        'Linear_Predicted': lr_mod_pred})
    
    
team_11_model_prediction.to_excel("Team_11_Model_Prediction.xlsx")
###############################################################################
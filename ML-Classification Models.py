# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:40:05 2019

@author: dangl
"""

###############################################################################
# Cleaning Data
###############################################################################


# Importing libraries
# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve


# This code will help extend the limit row and columns on console
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)


# Importing data
file = 'GOT_character_predictions.xlsx'
org_got = pd.read_excel(file)


org_got.info()
org_got.head()
org_got.shape
org_got.describe().round(2)

org_got.isnull().any()
print(org_got.isnull().sum())


# Filling median value for data of birth column
fill_dob = org_got['dateOfBirth'].median()
org_got['dateOfBirth'] = org_got['dateOfBirth'].fillna(fill_dob)


# Filling median for age column
fill_age = org_got['age'].median()
org_got['age'] = org_got['age'].fillna(fill_age)


# Flagging columns of missing value with binary value
for col in org_got:

    """ Create new columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if org_got[col].isnull().any():
        org_got['bin_'+col] = org_got[col].isnull().astype(int)
        
        
# Recalling new got dataset
got_new = org_got


got_new.corr()


print(got_new.isnull().sum())



"""
Since Random Forest cannot handle categorical variables, I will remove them all.
    In addition, all columns having missing values (except age) are categorical;
    therefore, I will remove them all plus age column. Age column has 1500 missing
    value, it will not affect alot on the model.
"""


# Creating feature set for Game of Throne
got_features = got_new.drop(['isAlive',
                         'name',
                         'title',
                         'culture',
                         'mother',
                         'father',
                         'heir',
                         'house',
                         'spouse',
                         'isAliveMother',
                         'isAliveFather',
                         'isAliveHeir',
                         'isAliveSpouse'],
axis=1)


got_features.isnull().any()  # No missing value for this feature set.


# Creating target set for Game of Throne
got_target = got_new.loc[:,'isAlive']


"""
By dropping columns in creating feature and target set, I have clean dataset.
"""


# Splitting data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(got_features,
                                                    got_target,
                                                    test_size=0.1,
                                                    random_state=508,
                                                    stratify = got_target)


# Checking if the ratio of splitting is good
y_train.value_counts()
y_train.sum() / y_train.count()

y_test.value_counts()
y_test.sum() / y_test.count()


###############################################################################
# Building decision tree on scikit learning
###############################################################################


# Create model object
c_tree = DecisionTreeClassifier(random_state = 508)


c_tree_fit = c_tree.fit(X_train, y_train)


print('Training Score', c_tree_fit.score(X_train, y_train).round(4))
print('Testing Score:', c_tree_fit.score(X_test, y_test).round(4))


""" The training score is overfit; therefore, we must optimize model with
GridSearchCV by seeking 2 values of max_depth and min_sample_leaf for my
DecisionTreeClassifier model.
"""


#####################################
# Optimizing model with GridSearchCV
#####################################


# Importing library
from sklearn.model_selection import GridSearchCV


#######################
#Testing to find 2 best optimization
#######################


# Creating 2 hyperparameter grids
depth_space = pd.np.arange(1, 10)
leaf_space = pd.np.arange(1, 500)

param_grid = {'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space}


# Building the model object one more time with new name for hyperparameter
c_tree_hp = DecisionTreeClassifier(random_state = 508)


# Creating a GridSearchCV object for my model object
c_tree_hp_cv = GridSearchCV(c_tree_hp, param_grid, cv = 3)


# Fitting it to the training data
c_tree_hp_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", c_tree_hp_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", c_tree_hp_cv.best_score_.round(4))


#######################
# Building optimal model based on two new values above
#######################


c_tree_opt = DecisionTreeClassifier(random_state = 508,
                                     max_depth = 7,
                                     min_samples_leaf = 13)


c_tree_opt.fit(X_train, y_train)


#Score new this new Optimal Model
TraiScor_OptClassTree = c_tree_opt.score(X_train, y_train).round(4)
TestScor_OptClassTree = c_tree_opt.score(X_test, y_test).round(4)


# predicting response variable based on this model
ClassTree_pred = c_tree_opt.predict(X_test)


ClassTree_pred_prob = c_tree_opt.predict_proba(X_test)

# Cross Validation Score
ClTr_CVscore = cross_val_score(c_tree_opt, got_features, got_target, cv = 3)


print('\nAverage: ',
      pd.np.mean(ClTr_CVscore).round(5),
      '\nMinimum: ',
      min(ClTr_CVscore).round(5),
      '\nMaximum: ',
      max(ClTr_CVscore).round(5))


# Computing Auc Score
AUC_ClTr = roc_auc_score(y_test, ClassTree_pred_prob[:,1])



###############################################################################
#Building KNeighborsClassifier Model on Scikit Learn
###############################################################################


"""
I continue to use same splitting datas above.
"""


# Importing library for knn
from sklearn.neighbors import KNeighborsClassifier


# Doing same code as we do before for testing k value of number neighbor
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


#Plotting the accuracy for training and testing set
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()



# Looking for the highest test accuracy
print(test_accuracy)



# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)



# It looks like 7 neighbors is the most accurate
knn_clf = KNeighborsClassifier(n_neighbors = 7)



# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)


# Saving score for KNN Classification
TraiScore_knn = knn_clf_fit.score(X_train, y_train).round(4)
TestScore_knn = knn_clf_fit.score(X_test, y_test).round(4)


# Predicting on X_train data
knn_clf_predict = knn_clf_fit.predict(X_test)
knn_clf_predict_prob = knn_clf_fit.predict_proba(X_test)


# Cross Validation Score
knn_cv = cross_val_score(knn_clf, got_features, got_target, cv = 3)


print('\nAverage: ',
      pd.np.mean(knn_cv).round(5),
      '\nMinimum: ',
      min(knn_cv).round(5),
      '\nMaximum: ',
      max(knn_cv).round(5))


# Computing Auc Score
AUC_knn = roc_auc_score(y_test, knn_clf_predict_prob[:,1])


###############################################################################
# Logistic Regression Model on Scikit Learn
###############################################################################


# importing library for logistic regression on sklearn
from sklearn.linear_model import LogisticRegression


skl_logit = LogisticRegression(solver = 'lbfgs',
                            C = 1)


# Fititng data on training set
skl_logit_fit = skl_logit.fit(X_train, y_train)


# Predictions
skl_logit_pred = skl_logit_fit.predict(X_test)
skl_logit_pred_prob = skl_logit_fit.predict_proba(X_test)


# Let's compare the testing score to the training score.
print('Training Score', skl_logit.score(X_train, y_train).round(4))
print('Testing Score:', skl_logit.score(X_test, y_test).round(4))


# Saving score to compare later
TraiScore_skl_logit = skl_logit.score(X_train, y_train).round(4)
TestScore_skl_logit = skl_logit.score(X_test, y_test).round(4)


# Cross Validation Score
skl_cv = cross_val_score(skl_logit, got_features, got_target, cv = 3)


print('\nAverage: ',
      pd.np.mean(knn_cv).round(5),
      '\nMinimum: ',
      min(knn_cv).round(5),
      '\nMaximum: ',
      max(knn_cv).round(5))


# Computing Auc Score
AUC_skl_logit = roc_auc_score(y_test, skl_logit_pred_prob[:,1])


###############################################################################
# Building Random Forest on Scikit Learn
###############################################################################


# Building Random Forest Model by using either gini or entropy
ran_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)


# Fitting random forest model on training set
rf_gini_fit = ran_forest_gini.fit(X_train, y_train)


# Scoring the gini model
print('Training Score', rf_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', rf_gini_fit.score(X_test, y_test).round(4))


# Saving training and testing scores
TraiScore_RF = rf_gini_fit.score(X_train, y_train).round(4)
TestScore_RF = rf_gini_fit.score(X_test, y_test).round(4)


# Predicting and saving
RF_Pred = rf_gini_fit.predict(X_test)
RF_Pred_Prob = rf_pred_prob = rf_gini_fit.predict_proba(X_test)


# Cross Validation Score
RF_CvScore = cross_val_score(rf_gini_fit,
                                     got_features,
                                     got_target,
                                     cv = 3)


print('\nAverage: ',
      pd.np.mean(RF_CvScore).round(5),
      '\nMinimum: ',
      min(RF_CvScore).round(5),
      '\nMaximum: ',
      max(RF_CvScore).round(5))


# Saving max value of cv score
RF_cv_max = max(RF_CvScore).round(5)


##########################################
# Building ROC Curve and compute AUC Score
##########################################


# Install library for auc and roc curve
from sklearn.metrics import roc_auc_score


# Plotting ROC CURVE
fig = plt.figure(figsize=(10, 10))
plt.plot(*roc_curve(y_test, rf_pred_prob[:, 1])[:2])
plt.legend(["Random Forest Classifier"], loc="upper left")
plt.plot((0., 1.), (0., 1.), "--k", alpha=.7) 
plt.xlabel("False Positive Rate", labelpad=20)
plt.ylabel("True Positive Rate", labelpad=20)
plt.title("ROC Curves", fontsize=10)
plt.show()


# Computing AUC score
AUC_rf = roc_auc_score(y_test, rf_pred_prob[:, 1]).round(5)


# Printing the final core for AUC and CV Score Max
print('The Max Score of CV: ',
      RF_cv_max,
      'and',
      'The AUC Score: ',
      AUC_rf)


# Creating dataframe for my prediction along with actual
Dang_Le_Pred = pd.DataFrame({'Actual' : y_test,
                                        'RF Predic': RF_Pred})


# Exporting data to local drive
Dang_Le_Pred.to_excel("Dang_Le_Prediction_GOT.xlsx")




def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')
        
plot_feature_importances(rf_gini_fit,
                         train = X_train,
                         export = False)



# let drop some unimportant features
got_features = got_new.drop(['isAlive',
                         'name',
                         'title',
                         'culture',
                         'mother',
                         'father',
                         'heir',
                         'house',
                         'spouse',
                         'isAliveMother',
                         'isAliveFather',
                         'isAliveHeir',
                         'isAliveSpouse',
                         'bin_isAliveHeir',
                         'bin_isAliveMother',
                         'bin_heir',
                         'bin_mother'],
axis=1)



X_train, X_test, y_train, y_test = train_test_split(got_features,
                                                    got_target,
                                                    test_size=0.1,
                                                    random_state=508,
                                                    stratify = got_target)





###############################################################################
# Building Gradiant Boosting Machine
###############################################################################


"""
I will run thi gradiant boosting model with GridSearchCV to optimize the
model.
"""


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


# Building a weak learner gbm
gbm_got = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1,
                                  n_estimators = 500,
                                  max_depth = 1,
                                  criterion = 'mse',
                                  warm_start = False,
                                  random_state = 508,
                                  )


gbm_got_fit = gbm_got.fit(X_train, y_train)


gbm_basic_predict = gbm_got_fit.predict(X_test)


gbm_basic_predict_prob = gbm_got_fit.predict_proba(X_test)


# Training and Testing Scores
print('Training Score', gbm_got_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_got_fit.score(X_test, y_test).round(4))



# Computing AUC score
AUC_gbm = roc_auc_score(y_test, gbm_basic_predict_prob[:, 1]).round(5)



def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')
        
plot_feature_importances(gbm_got_fit,
                         train = X_train,
                         export = False)






# Creating a hyperparameter grid
learn_space = pd.np.arange(0.1, 1.6, 0.1)
estimator_space = pd.np.arange(50, 250, 50)
depth_space = pd.np.arange(1, 10)
criterion_space = ['friedman_mse', 'mse', 'mae']


param_grid = {'learning_rate' : learn_space,
              'max_depth' : depth_space,
              'criterion' : criterion_space,
              'n_estimators' : estimator_space}


# Building Model Object
gbm_got_test = GradientBoostingClassifier(random_state = 508)


# Creating a GridSearchCV object
gbm_got_gridcv = GridSearchCV(gbm_got_test, param_grid, cv = 3)



# Fit it to the training data
gbm_got_gridcv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_got_gridcv.best_params_)
print("Tuned GBM Accuracy:", gbm_got_gridcv.best_score_.round(4))



###############################################################################
#Building Logistic Regression for Classification mode on statsmodels
###############################################################################


"""
I will use the same feature set and target set from above since it would be
good for non categorical columns.

"""


# Importing libraries
import statsmodels.formula.api as smf


# Concating X_train and y_train into DataFrame
got_df_train = pd.concat([X_train, y_train], axis=1)


got_df_train.info()
got_df_train.corr().round(3)


# Renaming S.No column since smf cannot read S.No
got_df_train.rename(columns={'S.No':'SNo'}, 
                 inplace=True)


# Building full model (non categorical features)
logistic_got = smf.logit(formula = """isAlive ~ SNo +
                                                male  +
                                                book1_A_Game_Of_Thrones +
                                                book2_A_Clash_Of_Kings +
                                                book3_A_Storm_Of_Swords +
                                                book4_A_Feast_For_Crows +
                                                book5_A_Dance_with_Dragons +
                                                isMarried +
                                                isNoble +
                                                age +
                                                numDeadRelations +
                                                popularity +
                                                bin_title +
                                                bin_culture +
                                                bin_mother +
                                                bin_father +
                                                bin_heir +
                                                bin_house +
                                                bin_isAliveMother +
                                                bin_isAliveFather +
                                                bin_isAliveHeir +
                                                bin_isAliveSpouse""",
                                                data = got_df_train)


# Fitting logistic_got
logistic_got_fit = logistic_got.fit()


logistic_got_fit.summary()


logistic_got_fit.pvalues


dir(logistic_got_fit)


print('AIC:', logistic_got_fit.aic.round(2))
print('BIC:', logistic_got_fit.bic.round(2))


"""
Statsmodels: score is not available for this library, how can we know if this
    model is good or not?
"""






































# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:15:05 2019

@author: Dang Nhat Le

Unsupervising Assignment: Mobile App
"""


###############################################################################
# Importing, splitting data
###############################################################################



# Importing known libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Importing new libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# import dataset
mob_app = pd.read_excel('finalExam_preparation_data.xlsx')



# Exploring datase
mob_app.describe()
mob_app.info()
mob_app.columns
mob_app.shape


###############################################################################
# PCA Method
###############################################################################

###############################################
# Step 1: Remove non-demographic data
###############################################
mob_app_drop = mob_app.iloc[:, 2:36]

mob_app_factors = mob_app_drop.drop(['q2r10'], axis = 1)    # thí há only value 0, then I terminate this column



"""
I also drop question 24, 25, and 26 since those question are like self evaluation.
"""


###############################################
# Step 2: Scale all new variables to equal variance
###############################################



scaler_mob = StandardScaler()


scaler_mob.fit(mob_app_factors)


MOB_Scaled = scaler_mob.transform(mob_app_factors)



########################
# Step 3: Run PCA without limiting the number of components
########################

mob_app_pca = PCA(n_components = None,
                           random_state = 508)


mob_app_pca.fit(MOB_Scaled)


PCA_Mob = mob_app_pca.transform(MOB_Scaled)



########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(mob_app_pca.n_components_)


plt.plot(features,
         mob_app_pca.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Mobile Apps Survey Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()



"""
Dang Le:
    I will choose only 5 components on the left side because their distance is
    far and easy to be explained while the rest on the right side is very closed
    indicating that their differences are not far; in other words, they are very
    similar.
"""



########################
# Step 5: Run PCA again based on the desired number of components
########################

PCA_Mob_New = PCA(n_components = 3,
                           random_state = 508)


PCA_Mob_New.fit(MOB_Scaled)



########################
# Step 6: Analyze factor loadings to understand principal components
########################

factor_mob_df = pd.DataFrame(pd.np.transpose(PCA_Mob_New.components_))


FACTOR_mob_df = factor_mob_df.set_index(mob_app_factors.columns[:]).round(3)


print(FACTOR_mob_df)



########################
# Step 7: Analyze factor strengths per survey taker
########################

MOB_pca_df = pd.DataFrame(PCA_Mob_New.transform(MOB_Scaled))


"""
Dataframe above shows how each survey taker responses to each component (group).
"""



########################
# Step 8: Rename your principal components and reattach demographic information
########################

MOB_pca_df.columns = ['seg 1',
                            'seg 2',
                            'seg 3']


final_pca_mob_df = pd.concat([mob_app.loc[ : , ['caseID', 'q1',
                                                'q48', 'q49',
                                                'q54', 'q56']],
                                              MOB_pca_df], axis = 1)



# Setting name for demographic informations
final_pca_mob_df = final_pca_mob_df.rename(index=str, columns = {'q1':'age',
                                                      'q48':'edu',
                                                      'q49':'status',
                                                      'q54':'race',
                                                      'q56':'income'})


"""
I will not reuse the question 50 since it has multiple columns; In addition,
    gender, children and hispanic/latino information will not be used. Because"
    Children has multiple columns and not neccessary.
    Hispanic could be included in race column as other race.
    Gender should not be a big matter for developing an app.
    
"""


# Renaming ages
age_names = {1 : 'Un 18',
                 2 : '18-24',
                 3 : '25-29',
                 4 : '30-34',
                 5 : '35-39',
                 6 : '40-44',
                 7 : '45-49',
                 8 : '50-54',
                 9 : '55-59',
                 10 : '60-64',
                 11 : '65&over'}


final_pca_mob_df['age'].replace(age_names, inplace = True)



# Renaming eduation column
edu_names = {1 : 'higschl',
                 2 : 'highschlgrad',
                 3 : 'college',
                 4 : 'collgrad',
                 5 : 'pgradstudy',
                 6 : 'pgraddegree'}


final_pca_mob_df['edu'].replace(edu_names, inplace = True)



# Renaming maritial status
sta_names = {1 : 'married',
                 2 : 'single',
                 3 : 'sig-partner',
                 4 : 'wid/divor'}


final_pca_mob_df['status'].replace(sta_names, inplace = True)



# Renaming race
race_names = {1 : 'White',
                 2 : 'Black',
                 3 : 'Asian',
                 4 : 'hawai/OPI',
                 5 : 'AmIndian',
                 6 : 'Other'}


final_pca_mob_df['race'].replace(race_names, inplace = True)



# Renaming income
income_names = {1 : '<10k',
                 2 : '10-14',
                 3 : '15-19',
                 4 : '20-29',
                 5 : '30-39',
                 6 : '40-49',
                 7 : '50-59',
                 8 : '60-69',
                 9 : '70-79',
                 10 : '80-89',
                 11 : '90-99',
                 12 : '100-124',
                 13 : '125-149',
                 14 : '>150k'}


final_pca_mob_df['income'].replace(income_names, inplace = True)


#########################
# Analyzing team 1
#########################


# Analyzing age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y =  'seg 1',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


# Analyzing status
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'status',
            y =  'seg 1',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


# Analyzing education
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'edu',
            y =  'seg 1',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


# Analyzing race
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'race',
            y =  'seg 1',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


# Analyzing income
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'income',
            y =  'seg 1',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()



#########################
# Analyzing team 2
#########################

# Analyzing age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y =  'seg 2',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


# Analyzing status
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'status',
            y =  'seg 2',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


# Analyzing education
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'edu',
            y =  'seg 2',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


# Analyzing race
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'race',
            y =  'seg 2',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


# Analyzing income
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'income',
            y =  'seg 2',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()



#########################
# Analyzing team 3
#########################

# Analyzing age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y =  'seg 3',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


# Analyzing status
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'status',
            y =  'seg 3',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


# Analyzing education
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'edu',
            y =  'seg 3',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


# Analyzing race
fig, ax = plt.subplots(figsize = (8, 6))
sns.boxplot(x = 'race',
            y =  'seg 3',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


# Analyzing income
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'income',
            y =  'seg 3',
            data = final_pca_mob_df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()


































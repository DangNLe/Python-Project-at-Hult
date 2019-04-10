# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 22:42:19 2019

@author: dangl
"""




# Importing known libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Importing new libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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

mob_app_factors = mob_app_drop.drop(['q2r10'], axis = 1)    # this has only value 0, then I terminate this column


######################################
# Scaling data again for cluster
######################################


scaler_cluster = StandardScaler()


scaler_cluster.fit(mob_app_factors)


Cluster_Scaled = scaler_cluster.transform(mob_app_factors)



########################
# Experiment with different numbers of clusters
########################

mob_app_k = KMeans(n_clusters = 5,
                      random_state = 508)


mob_app_k.fit(mob_app_factors)


mob_kmeans_clusters = pd.DataFrame({'cluster': mob_app_k.labels_})


print(mob_kmeans_clusters.iloc[: , 0].value_counts())



########################
# Analyze cluster centers
########################

cent_mob = mob_app_k.cluster_centers_


cent_mob_df = pd.DataFrame(cent_mob)


# Renaming columns
cent_mob_df.columns = mob_app_factors.columns

print(cent_mob_df)


# Sending data to Excel
cent_mob_df.to_excel('Mob_Cluster_Centroids.xlsx')


########################
# Analyze cluster memberships
########################


Mob_app_df = pd.DataFrame(Cluster_Scaled)


Mob_app_df.columns = mob_app_factors.columns


clusters_mob_df = pd.concat([mob_kmeans_clusters,
                         Mob_app_df],
                         axis = 1)


print(clusters_mob_df)



#############################################
# Reattaching demographic columns
#############################################


# factor data only
mob_app_factors


# Concatting datasets
final_clusters_mob_df = pd.concat([mob_app.loc[ : , ['caseID', 'q1',
                                                'q48', 'q49',
                                                'q54', 'q56',
                                                'q55']],
                                              clusters_mob_df], axis = 1)


# Setting name for demographic informations
final_clusters_mob_df = final_clusters_mob_df.rename(index=str,
                                                     columns = {'q1':'age',
                                                      'q48':'edu',
                                                      'q49':'status',
                                                      'q54':'race',
                                                      'q55':'His/Lati',
                                                      'q56':'income'})


"""
I will not reuse the question 50 since it has multiple columns; In addition,
    gender, children and hispanic/latino information will not be used. Because"
    Children has multiple columns and not neccessary.
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


final_clusters_mob_df['age'].replace(age_names, inplace = True)



# Renaming eduation column
edu_names = {1 : 'higschl',
                 2 : 'highschlgrad',
                 3 : 'college',
                 4 : 'collgrad',
                 5 : 'pgradstudy',
                 6 : 'pgraddegree'}


final_clusters_mob_df['edu'].replace(edu_names, inplace = True)



# Renaming maritial status
sta_names = {1 : 'married',
                 2 : 'single',
                 3 : 'sig-partner',
                 4 : 'wid/divor'}


final_clusters_mob_df['status'].replace(sta_names, inplace = True)



# Renaming race
race_names = {1 : 'White',
                 2 : 'Black',
                 3 : 'Asian',
                 4 : 'hawai/OPI',
                 5 : 'AmIndian',
                 6 : 'Other'}


final_clusters_mob_df['race'].replace(race_names, inplace = True)



# Renaming hispanic/latino
his_lati_names = {1 : 'Yes',
                 2 : 'No'}


final_clusters_mob_df['His/Lati'].replace(his_lati_names, inplace = True)



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


final_clusters_mob_df['income'].replace(income_names, inplace = True)

Final_Clts_Df = final_clusters_mob_df




#########################
# Analyzing Demographic
#########################


# Analyzing age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'age',
            y =  'q2r1',
            hue = 'cluster',
            data = Final_Clts_Df)

plt.ylim(-10, 9)
plt.tight_layout()
plt.show()














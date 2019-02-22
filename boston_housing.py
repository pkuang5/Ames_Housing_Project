import numpy as np
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

#reades and imports train and test dataframe 
train = pd.read_csv("all/train.csv")
test = pd.read_csv("all/test.csv")

#makes copy of  dataframes
train_copy = train
test_copy = test

#makes an array of the name of all attributes
names = list(train)

names_test = list(train)
names_test.pop(80)

#locates deletes unimportant attributes
deleted_attribute = [0,2,5,6,7,9,10,11,14,15,21,22,23,24,27,28,31,32,33,35,36,
                     37,39,40,42,45,47,48,50,51,52,53,55,57,58,60,63,64,66,67,
                     68,69,70,71,72,73,74,75,76,77,78,80]
for j in reversed(deleted_attribute): #deletes unimportant attributes
    names.pop(j)

#array of attributues that need lienar regression
linear_attribute = [1,2,9,10,12,15,16,18,19,20,24,26] 


#filling in empty values with median values for all numerical attributes  in train and test dataframes 
for i in names_test:
    if train_copy[i].dtypes=='int64' or train_copy[i].dtypes=='float64':
        train_copy[i] = train_copy[i].fillna(train_copy[i].median())
    if test[i].dtypes=='int64' or test[i].dtypes=='float64':
        test_copy[i] = test_copy[i].fillna(test_copy[i].median())
            
#converting object attrbutes to int type
for i in names_test:
    if train_copy[i].dtypes=="O":
        train_copy[i] = train_copy[i].astype('category')
        train_copy[i] = train_copy[i].cat.codes
for i in names_test:
    if test[i].dtypes=="O":
        test[i] = test[i].astype('category')
        test[i] = test[i].cat.codes
        
#for i in names:
#    for k in linear_attribute:
#        if i==names[k]:
#            z_array = np.abs(stats.zscore(train_copy[i])) #creates an array z that holds the z-score (SD from mean
#            z_outlier_indexes = np.where(z_array > 3) #creates an nested array of the indexes of the outliers (z-score>3)
#            z_outlier_indexes = z_outlier_indexes[0] #converts nested array into an array
#            train_copy = train_copy.drop(z_outlier_indexes) #creates train_copy which omits the outliers


#for i in names:
#    print_linear = False;
#    for k in linear_attribute:
#        if i==names[k]:
#            print_linear = True;
#    if print_linear==True:        
#        plt.figure()
#        ax = sns.regplot(x=train_copy[i], y=train_copy["SalePrice"]) 
#    else:
#        plt.figure()
#        ax = sns.stripplot(x=train_copy[i], y=train_copy["SalePrice"]) 
        
#modeling via decision tree

      
#creating target set (Sale Price)   
target = train_copy["SalePrice"].values

#creating features  set (all attributes except Sale Price)
features = train_copy[names].values
        
## Fit your first decision tree: my_tree
#my_tree = tree.DecisionTreeClassifier()
#my_tree = my_tree.fit(features, target)
## Look at the importance and score of the included features
##print(my_tree.feature_importances_)
##print(my_tree.score(features, target))
#
##Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
#max_depth = 10
#min_samples_split = 5
#my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
#my_tree_two = my_tree_two.fit(features, target)
## Look at the importance and score of the included features
#print(my_tree_two.feature_importances_)
#print(my_tree_two.score(features, target))
#
#
## Extract the features from the test set
#test_features = test[names].values
#
## Make your prediction using the test set
#my_prediction = my_tree_two.predict(test_features)
#
## Create a data frame with two columns: Id & SalePrice. SalePrice contains your predictions
#Id =np.array(test["Id"]).astype(int)
#my_solution = pd.DataFrame(my_prediction, Id, columns = ["SalePrice"])
#print(my_solution)
#
## Check that your data frame has 418 entries
#print(my_solution.shape)
#
## Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_one.csv", index_label = ["Id"])

features_forest = train[names]
forest  = RandomForestClassifier(max_depth = 10, min_samples_split = 2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

print(my_forest.score(features_forest, target))

test_features = test[names].values
pred_forest = my_forest.predict(test_features)

# Create a data frame with two columns: Id & SalePrice. SalePrice contains your predictions
Id =np.array(test["Id"]).astype(int)
my_solution = pd.DataFrame(pred_forest, Id, columns = ["SalePrice"])
print(my_solution)
# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_two.csv", index_label = ["Id"])


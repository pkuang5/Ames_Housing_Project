import numpy as np
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


train = pd.read_csv("all/train.csv")
#print(train.head())
test = pd.read_csv("all/test.csv")
#print(test.head())

train_copy = train
test_copy = test

names = list(train)
plt.matshow(train.corr())
#locates deletes unimportant attributes
deleted_attribute = [0,2,5,6,7,9,10,11,14,15,21,22,23,24,27,28,31,32,33,35,36,
                     37,39,40,42,45,47,48,50,51,52,53,55,57,58,60,63,64,66,67,
                     68,69,70,71,72,73,74,75,76,77,78,80]
for j in reversed(deleted_attribute): #deletes unimportant attributes
    names.pop(j)

#array of attributues that need lienar regression
linear_attribute = [1,2,9,10,12,15,16,18,19,20,24,26] 


for i in names:
    for k in linear_attribute:
        if i==names[k]:
            train_copy[i] = train_copy[i].fillna(train_copy[i].median())
        

train_copy2 = train_copy

for i in names:
    for k in linear_attribute:
        if i==names[k]:
            z_array = np.abs(stats.zscore(train_copy[i])) #creates an array z that holds the z-score (SD from mean
            z_outlier_indexes = np.where(z_array > 3) #creates an nested array of the indexes of the outliers (z-score>3)
            z_outlier_indexes = z_outlier_indexes[0] #converts nested array into an array
            train_copy2 = train_copy.drop(z_outlier_indexes) #creates train_copy2 which omits the outliers


for i in names:
    print_linear = False;
    for k in linear_attribute:
        if i==names[k]:
            print_linear = True;
    if print_linear==True:        
        plt.figure()
        ax = sns.regplot(x=train_copy2[i], y=train_copy2["SalePrice"]) 
    else:
        plt.figure()
        ax = sns.stripplot(x=train_copy2[i], y=train_copy2["SalePrice"]) 
        
print(train)
print(train_copy2)



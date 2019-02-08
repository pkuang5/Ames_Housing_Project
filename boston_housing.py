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

#locates deletes unimportant attributes
deleted_attribute = [0,2,5,6,7,9,10,11,14,15,21,22,23,24,27,28,31,32,33,35,36,
                     37,39,40,42,45,47,48,50,51,52,53,55,57,58,60,63,64,66,67,
                     68,69,70,71,72,73,74,75,76,77,78,80]
for j in reversed(deleted_attribute): #deletes unimportant attributes
    names.pop(j)

#array of attributues that need lienar regression
linear_attribute = [1,2,9,10,12,15,16,18,19,20,24,26] 




#for i in names:
#z_array = np.abs(stats.zscore(train[train['LotFrontage'].isnull(), "LotFrontage"])) #creates an array z that holds the z-score (SD from mean)
#z_array = np.abs(stats.zscore(~train[train['LotFrontage'].isnull(), "LotFrontage"])) #creates an array z that holds the z-score (SD from mean)
#
#z_outlier_indexes = np.where(z_array > 3) 
#z_outlier_indexes = z_outlier_indexes[0]
#train_copy = train.drop(z_outlier_indexes)
#print(z_array)
#print(z_outlier_indexes)
#    


 
train['LotFrontage'].isnull()
train['GrLivArea'].isnull().sum()

print(train.loc[~train['LotFrontage'].isnull()])



import numpy as np
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv("all/train.csv")
print(train.head())
test = pd.read_csv("all/test.csv")
print(test.head())

names = list(train)
print(len(names))
deleted_attribute = [0,2,5,6,7,9,10,11,14,15,21,22,23,24,27,28,31,32,33,35,36,
                     37,39,40,42,45,47,48,50,51,52,53,55,57,58,60,63,64,66,67,
                     68,69,70,71,72,73,74,75,76,77,78,80]

for j in reversed(deleted_attribute):
    names.pop(j)

linear_attribute = [1,2,9,10,12,15,16,18,19,20,24,26]

for i in names:
    print_linear = False;
    for k in linear_attribute:
        if i==names[k]:
            print_linear = True;
    if print_linear==True:        
        plt.figure()
        ax = sns.regplot(x=train[i], y=train["SalePrice"]) 
    else:
        plt.figure()
        ax = sns.stripplot(x=train[i], y=train["SalePrice"]) 

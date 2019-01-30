import numpy as np
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv("all/train.csv")
print(train.head())
test = pd.read_csv("all/test.csv")
print(test.head())
#plt.figure()
#ax = sns.scatterplot(x = train["OverallQual"], y = train["SalePrice"]).set_title("Overall Quality bitch")
#plt.figure()
#ax = sns.stripplot(x=train["MSSubClass"], y=train["SalePrice"]) #attribute not tht important
#plt.figure()
#ax = sns.stripplot(x=train["MSSubClass"], y=train["SalePrice"]) 
names = list(train)

for i in names:
    plt.figure()
    ax = sns.stripplot(x=train[i], y=train["SalePrice"]) 


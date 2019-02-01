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

for i in names:
    plt.figure()
    ax = sns.stripplot(x=train[i], y=train["SalePrice"]) 
#Id 0
#LotFrontage on line  graph 3
#LotArea on line graph 4
#Street irrelevant 5
#Alley irrelevant 6
#LotShape irrevlevant 7
#LandContour 8

    
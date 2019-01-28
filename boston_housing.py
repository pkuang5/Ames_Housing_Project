import pandas as pd
train = pd.read_csv("all/train.csv")
print(train.head())
test = pd.read_csv("all/test.csv")
print(test.head())

train["Expensive"] = float('NaN')
train["Expensive"][train["SalePrice"] < train["SalePrice"].median()] = 0
train["Expensive"][train["SalePrice"] >= train["SalePrice"].median()] = 1
#print(train["Expensive"].value_counts(normalize = True))
print(train["Expensive"][train["MSSubClass"] == 20].value_counts(normalize = True))

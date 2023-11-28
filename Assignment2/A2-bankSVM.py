from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


bankData = pd.read_csv('A2-bank/bank-additional.csv', sep=';')

X = bankData.iloc[:,:-1].values
#print(X)


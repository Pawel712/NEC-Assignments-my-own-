# This python code is used to clean the bank-additional dataset

import pandas as pd


df = pd.read_csv('A2-bank/bank-additional.csv', sep=';')

#Changing the y column to 0 and 1
df['y'] = df['y'].replace({'no': 0, 'yes': 1})

# Remove every row that has more than 1 unknowns values
df = df[df.isin(['unknown']).sum(axis=1) <= 1]

# For these columns: job, marital and education, remove rows that has unknown values
df = df[df['job'] != 'unknown']
df = df[df['marital'] != 'unknown']
df = df[df['education'] != 'unknown']

#----- Weighted mean ------ #

#number of rows:
rows = df.shape[0]
professions = ["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed"]

for i in professions:
    n = df['job'].value_counts()[i] # amount of a certain profession in job column
    onesInY = df[(df['job'] == i) & (df['y'] == 1)].shape[0] # calculate amount of values "1" in y that have "admin." in job column
    optionMean = onesInY / n
    overallMean = onesInY / rows
    weightedMean = ((optionMean * n) + (n*overallMean))/(n+n)
    df['job'] = df['job'].replace(i, weightedMean) # replace all the values "admin." in job column with weighted mean

#----- Weighted mean ------ #


df.to_csv("ModifiedDatasets/bank-additionalModified.csv", sep=";", index=False)

# This python code is used to clean the bank-additional dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def removeUnknownValues(df):
    # Remove every row that has more than 1 unknowns values
    df = df[df.isin(['unknown']).sum(axis=1) <= 1]

    # For these columns remove rows that has unknown values
    df = df[df['job'] != 'unknown']
    df = df[df['marital'] != 'unknown']
    df = df[df['education'] != 'unknown']
    df = df[df['housing'] != 'unknown']
    df = df[df['loan'] != 'unknown']

    return df

def ConvertColumnValuesToOneAndTwo(df):
    df['y'] = df['y'].replace({'no': 0, 'yes': 1})
    df['housing'] = df['housing'].replace({'no': 0, 'yes': 1})
    df['loan'] = df['loan'].replace({'no': 0, 'yes': 1})
    df['contact'] = df['contact'].replace({'cellular': 0, 'telephone': 1})
    return df

def weightedMean(df):
    rows = df.shape[0] #number of rows:
    columnsToChange = ['job', 'marital', 'education']
    professions = ["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed"]
    maritals = ["divorced","married","single"]
    educations = ["basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree"]
    months = ['may', 'jun', 'nov', 'sep', 'jul', 'aug', 'mar', 'oct', 'apr', 'dec']
    dayOfWeeks = ['fri', 'wed', 'mon', 'thu', 'tue']

    for i in professions:
        n = df['job'].value_counts()[i] # amount of a certain profession in job column
        onesInY = df[(df['job'] == i) & (df['y'] == 1)].shape[0] # calculate amount of values "1" in y that have "admin." in job column
        optionMean = onesInY / n
        overallMean = onesInY / rows
        weightedMean = ((optionMean * n) + (n*overallMean))/(n+n)
        df['job'] = df['job'].replace(i, weightedMean) # replace all the values "admin." in job column with weighted mean

    for i in maritals:
        n = df['marital'].value_counts()[i] # amount of a certain marital status in marital column
        onesInY = df[(df['marital'] == i) & (df['y'] == 1)].shape[0] # calculate amount of values "1" in y that have "admin." in marital column
        optionMean = onesInY / n
        overallMean = onesInY / rows
        weightedMean = ((optionMean * n) + (n*overallMean))/(n+n)
        df['marital'] = df['marital'].replace(i, weightedMean) # replace all the values "admin." in marital column with weighted mean

    for i in educations:
        n = df['education'].value_counts()[i] # amount of a certain education in education column
        onesInY = df[(df['education'] == i) & (df['y'] == 1)].shape[0] # calculate amount of values "1" in y that have "admin." in education column
        optionMean = onesInY / n
        overallMean = onesInY / rows
        weightedMean = ((optionMean * n) + (n*overallMean))/(n+n)
        df['education'] = df['education'].replace(i, weightedMean) # replace all the values "admin." in education column with weighted mean

    for i in months:
        n = df['month'].value_counts()[i] # amount of a certain month in month column
        onesInY = df[(df['month'] == i) & (df['y'] == 1)].shape[0] # calculate amount of values "1" in y that have "admin." in month column
        optionMean = onesInY / n
        overallMean = onesInY / rows
        weightedMean = ((optionMean * n) + (n*overallMean))/(n+n)
        df['month'] = df['month'].replace(i, weightedMean) # replace all the values "admin." in month column with weighted mean

    for i in dayOfWeeks:
        n = df['day_of_week'].value_counts()[i] # amount of a certain day in day column
        onesInY = df[(df['day_of_week'] == i) & (df['y'] == 1)].shape[0] # calculate amount of values "1" in y that have "admin." in day column
        optionMean = onesInY / n
        overallMean = onesInY / rows
        weightedMean = ((optionMean * n) + (n*overallMean))/(n+n)
        df['day_of_week'] = df['day_of_week'].replace(i, weightedMean) # replace all the values "admin." in day column with weighted mean
    return df

def oneHotEncoding(df):
    uniqueValuesdefault = df['default'].unique().tolist()

    for i in uniqueValuesdefault: # creating new columns for the unique values of default and adding 0 and 1
        df[i] = (df['default'] == i).astype(int)

    df.rename(columns={'unknown': 'unknownDefault'}, inplace=True) # changes name of unknown column to unknownDefault
    df.rename(columns={'no': 'NoSuccess'}, inplace=True)
    df.rename(columns={'yes': 'YesSuccess'}, inplace=True)

    df.drop('default', axis=1, inplace=True)

    #column poutcome:
    unique_values_poutcome = df['poutcome'].unique().tolist()

    for i in unique_values_poutcome: # creating new columns for the unique values of poutcome and adding 0 and 1
        df[i] = (df['poutcome'] == i).astype(int)

    df.rename(columns={'success': 'successPoutcome'}, inplace=True)
    df.rename(columns={'failure': 'failurePoutcome'}, inplace=True)
    df.rename(columns={'nonexistent': 'nonexistentPoutcome'}, inplace=True)

    df.drop('poutcome', axis=1, inplace=True) 


    df = df[[col for col in df.columns if col != 'y'] + ['y']]#move the y column to be last
    return df

def normalization(df):
    columnsToBeNormalizedMinMax = ['age', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    scaler = MinMaxScaler()

    for i in columnsToBeNormalizedMinMax:
        df[i] = scaler.fit_transform(df[[i]])
    # z-score normalization for column pdays:
    df['pdays'] = (df['pdays'] - df['pdays'].mean()) / df['pdays'].std() 
    return df

with open('../A2-bank/bank-additional.csv', 'r') as file:
    df = pd.read_csv('../A2-bank/bank-additional.csv', sep=';')

df = removeUnknownValues(df)
df = ConvertColumnValuesToOneAndTwo(df)
df = weightedMean(df)
df = oneHotEncoding(df)
df = normalization(df)

df.drop('duration', axis=1, inplace=True)

df.to_csv("../ModifiedDatasets/bank-additionalModified.csv", sep=";", index=False)

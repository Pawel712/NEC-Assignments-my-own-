from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def normalizingColumns(df):
    columnsNormalizeMinMax = ['age', 'pressurehigh', 'pressurelow']
    columnsNormaluzeZ_score = ['impluse', 'glucose', 'kcm', 'troponin']
    
    scaler = MinMaxScaler()
    for i in columnsNormalizeMinMax:
        df[i] = scaler.fit_transform(df[[i]])

    for i in columnsNormaluzeZ_score:
        df[i] = (df[i] - df[i].mean()) / df[i].std() 

    return df

def ConvertToBinaryValues(df):
    df['class'] = df['class'].replace({'negative': 0, 'positive': 1})
    return df

with open('../dataset3/heartAttack.csv') as file:
    df = pd.read_csv('../dataset3/heartAttack.csv', sep=',')

df = normalizingColumns(df)
df = ConvertToBinaryValues(df)

df.to_csv("../ModifiedDatasets/HeartAttackModified.csv", sep=";", index=False)


# columns of heartAttack dataset
#print(df.columns)
#columns = ['age', 'gender', 'impluse', 'pressurehigh', 'pressurelow', 'glucose',
#       'kcm', 'troponin', 'class']
##check outliers:
#for i in columns:
#    print(i)
#    #print("Unique values of the column: ", i , df[i].unique())
#    print("Max and min value: of column: ", i, df[i].max(), df[i].min())

#unique_values = df['gender'].unique()

#maxValue = df['gender'].max()
#minValue = df['gender'].min()
#print("maxValue: ", maxValue)
#print("minValue: ", minValue)




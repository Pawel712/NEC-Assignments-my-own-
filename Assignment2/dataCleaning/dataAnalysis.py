import pandas as pd


with open('../A2-bank/bank-additional.csv', 'r') as file:
    df = pd.read_csv('../A2-bank/bank-additional.csv', sep=';')

# Count the number of a specific value in column loan
#print(df['emp.var.rate'].value_counts())

#print the values from the column duration
#print("Duration: ",df['duration'])

#print the values in column duration that has value 0
#print(df[df['duration'] == 0]['duration'])

#print the values from smallest to biggest in column duration
#print("Sorted values: ",df['poutcome'].sort_values())

#print the column unique values and convert them to a list
#unique_values = df['poutcome'].unique().tolist()
#print(unique_values)


print(len(df.columns))



import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#reading the csv file of the crimes in the boston.
data=pd.read_csv(r"C:\\Users\\patel\\Documents\\CS-5100 Foundation To Artificial Intelligence\\Project-work\\archive\\crime.csv",encoding='latin-1')
#reading the the first 10 files of the csv.
print(data.head(10))
#reading the offense of crimes.
offense=pd.read_csv(r"C:\Users\patel\Documents\CS-5100 Foundation To Artificial Intelligence\Project-work\archive\offense_codes.csv",encoding='latin-1')
#reading the first 10 files of the head of the 
print(offense.head(10))
#description of the data of the crimes in the boston. 
print(data.describe())
#description of the offense code in the boston.
print(offense.describe())

feature_column = 'Feature'

# Find the minimum and maximum values of the feature
min_value = data['OFFENSE_CODE'].min()
print(min_value)
max_value = data['OFFENSE_CODE'].max()

# Normalize the feature using Min-Max normalization
data['Normalized_Feature'] = (data['OFFENSE_CODE'] - min_value) / (max_value - min_value)

# Save the normalized data back to a CSV file
data.to_csv('normalized_crimefile.csv', index=False)
# Assuming your CSV has two columns named 'Feature1' and 'Feature2', adjust accordingly
X = data[['REPORTING_AREA', 'OFFENSE_CODE']].values


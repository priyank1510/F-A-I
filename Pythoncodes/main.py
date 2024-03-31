import pandas as pd
from sklearn.preprocessing import LabelEncoder

# reading the csv file of the crimes in the boston.
crimeData = pd.read_csv(
    r"C:\Users\patel\Documents\CS-5100 Foundation To Artificial Intelligence\Project-work\Pythoncodes\archive\crime.csv",
    encoding='latin-1')
# datatype of the crime data
print("file of the crime data: ", type(crimeData))
# reading  the first 10 files of the csv.
print(crimeData.head(10))
# reading the offense of crimes.
offense = pd.read_csv(
    r"C:\Users\patel\Documents\CS-5100 Foundation To Artificial Intelligence\Project-work\Pythoncodes\archive\offense_codes.csv",
    encoding='latin-1')
# reading the first 10 files of the head of the
print(offense.head(10))
# description of the data of the crimes in the boston.
print(crimeData.describe())
# description of the offense code in the boston.
print(offense.describe())

# checking the data types of the csv file.
print(crimeData.dtypes)

# Total columns in the crime data
columns = crimeData.columns
print("The columns of the crime data: \n", columns)

# Total number of crimes in each district
district_crime = crimeData['DISTRICT'].value_counts()
print("The total number of crimes in each district: \n", district_crime)

# Performing Exploratory Data Analysis
nullCount = crimeData.isnull().sum()
print("The values of the csv that are null: \n", nullCount)

# checking the shape of the csv file.
print("The shape of the data: ", crimeData.shape)

# Drop rows where 'DISTRICT' is NaN
crimeData = crimeData.dropna(subset=['DISTRICT'])

# checking the description of the data.
print(crimeData.describe())

# checking the null count after the drop of the data.
nullCount = crimeData.isnull().sum()
print("The values of the csv that are null: \n", nullCount)

# Drop rows where 'DISTRICT' is NaN
crimeData = crimeData.dropna(subset=['DISTRICT'])

# checking the description of the data.
print(crimeData.describe())

# Fill missing values for 'UCR_PART' and 'STREET' with mode
for column in ['UCR_PART', 'STREET']:
    crimeData[column].fillna(crimeData[column].mode()[0], inplace=True)

# Drop rows where 'STREET' is NaN
crimeData = crimeData.dropna(subset=['STREET'])

# checking the description of the data.
print(crimeData.describe())

# drop the location as lat and long is already given.
crimeData = crimeData.drop('Location', axis=1)
# now checking the description of the data.
print(crimeData.describe())
# checking whether the location is dropped or not.
print(crimeData.columns)

# checking the null count after the drop of the data.
nullCount = crimeData.isnull().sum()
print("The values of the csv that are null: \n", nullCount)

# Fill missing values for 'Lat' and 'Long' with mean
for column in ['Lat', 'Long']:
    crimeData[column].fillna(crimeData[column].mean(), inplace=True)

# Replace NaN values with 0 in 'SHOOTING' column
crimeData['SHOOTING'].fillna(0, inplace=True)

# Replace 'Y' with 1.4 in 'SHOOTING' column
crimeData['SHOOTING'].replace('Y', 1, inplace=True)



# checking the null values of the data.
nullCount = crimeData.isnull().sum()
print("The values of the csv that are null: \n", nullCount)

# converting categorical data to numerical data.
value_count = crimeData['DAY_OF_WEEK'].value_counts()
print("The value count of the DAY_OF_WEEK: \n", value_count)

# Save the cleaned data to a new CSV file
cleanCrimeData = crimeData.to_csv('crimeData_clean.csv', index=False)


# Create a dictionary to map days of the week to numerical values
days_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}

# Use the map function to replace the days of the week with numerical values
crimeData['DAY_OF_WEEK'] = crimeData['DAY_OF_WEEK'].map(days_mapping)
# Print some data.
print(crimeData['DAY_OF_WEEK'].head())

# Create a dictionary to map UCR_PART to numerical values
ucr_mapping = {'Part One': 1, 'Part Two': 2, 'Part Three': 3}
# Use the map function to replace the UCR_PART with numerical values
crimeData['UCR_PART'] = crimeData['UCR_PART'].map(ucr_mapping)
# Print some data.
print(crimeData['UCR_PART'].head())

# Create a label encoder
encode = LabelEncoder()
# Fit and transform the 'OFFENSE_CODE_GROUP' column
crimeData['OFFENSE_CODE_GROUP'] = encode.fit_transform(crimeData['OFFENSE_CODE_GROUP'])
crimeData['OFFENSE_DESCRIPTION'] = encode.fit_transform(crimeData['OFFENSE_DESCRIPTION'])
crimeData['STREET'] = encode.fit_transform(crimeData['STREET'])
# Print the first 5 rows to see the changes
print(crimeData.head())
print(crimeData['OFFENSE_DESCRIPTION'].head())

# now we will save the normalized data into the csv file so that it can be used for the further analysis.
# In the Machine Learning models.
newCrimeData= crimeData.to_csv('crimeData_normal.csv', index=False)

# feature_column = 'Feature'
#
# # Find the minimum and maximum values of the feature
# min_value = crimeData['OFFENSE_CODE'].min()
# print(min_value)
# max_value = crimeData['OFFENSE_CODE'].max()
#
# # Normalize the feature using Min-Max normalization
# crimeData['Normalized_Feature'] = (crimeData['OFFENSE_CODE'] - min_value) / (max_value - min_value)
# # Save the normalized data back to a CSV file
# crimeData.to_csv('normalized_crime file.csv', index=False)
# # Assuming your CSV has two columns named 'Feature1' and 'Feature2', adjust accordingly
# X = crimeData[['REPORTING_AREA', 'OFFENSE_CODE']].values

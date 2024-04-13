import numpy as np
import pandas as pd
from folium import folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
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
crimeNormalData = crimeData.dropna(subset=['DISTRICT'])

# checking the description of the data.
print(crimeNormalData.describe())

# checking the null count after the drop of the data.
nullCount = crimeNormalData.isnull().sum()
print("The values of the csv that are null: \n", nullCount)

# Drop rows where 'DISTRICT' is NaN
crimeNormalData = crimeNormalData.dropna(subset=['DISTRICT'])

# checking the description of the data.
print(crimeNormalData.describe())

# Fill missing values for 'UCR_PART' and 'STREET' with mode
for column in ['UCR_PART', 'STREET']:
    crimeNormalData[column].fillna(crimeNormalData[column].mode()[0], inplace=True)



# Drop rows where 'STREET' is NaN
crimeNormalData = crimeNormalData.dropna(subset=['STREET'])

# checking the description of the data.
print(crimeNormalData.describe())

# drop the location as lat and long is already given.
crimeNormalData = crimeNormalData.drop('Location', axis=1)
# now checking the description of the data.
print(crimeNormalData.describe())
# checking whether the location is dropped or not.
print(crimeNormalData.columns)

# checking the null count after the drop of the data.
nullCount = crimeNormalData.isnull().sum()
print("The values of the csv that are null: \n", nullCount)

# Fill missing values for 'Lat' and 'Long' with mean
# for column in ['Lat', 'Long']:
#     crimeNormalData[column].fillna(crimeNormalData[column].mean(), inplace=True)




# Drop rows where 'Lat' and 'Long' are 0
crimeNormalData = crimeNormalData[(crimeNormalData['Lat'] != 0) & (crimeNormalData['Long'] != 0)]










# Replace NaN values with 0 in 'SHOOTING' column
crimeNormalData['SHOOTING'].fillna(0, inplace=True)

# Replace 'Y' with 1.4 in 'SHOOTING' column
crimeNormalData['SHOOTING'].replace('Y', 1, inplace=True)

# checking the null values of the data.
nullCount = crimeNormalData.isnull().sum()
print("The values of the csv that are null: \n", nullCount)

# converting categorical data to numerical data.
value_count = crimeNormalData['DAY_OF_WEEK'].value_counts()
print("The value count of the DAY_OF_WEEK: \n", value_count)

# Save the cleaned data to a new CSV file
cleanCrimeData = crimeNormalData.to_csv('crimeData_clean.csv', index=False)

# Create a dictionary to map days of the week to numerical values
days_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}

# Use the map function to replace the days of the week with numerical values
crimeNormalData['DAY_OF_WEEK'] = crimeNormalData['DAY_OF_WEEK'].map(days_mapping)
# Print some data.
print(crimeNormalData['DAY_OF_WEEK'].head())

# Create a dictionary to map UCR_PART to numerical values
ucr_mapping = {'Part One': 1, 'Part Two': 2, 'Part Three': 3}
# Use the map function to replace the UCR_PART with numerical values
crimeNormalData['UCR_PART'] = crimeNormalData['UCR_PART'].map(ucr_mapping)
# Print some data.
print(crimeNormalData['UCR_PART'].head())

# Create a label encoder
encode = LabelEncoder()
# Fit and transform the 'OFFENSE_CODE_GROUP' column
crimeNormalData['OFFENSE_CODE_GROUP'] = encode.fit_transform(crimeNormalData['OFFENSE_CODE_GROUP'])
crimeNormalData['OFFENSE_DESCRIPTION'] = encode.fit_transform(crimeNormalData['OFFENSE_DESCRIPTION'])
crimeNormalData['INCIDENT_NUMBER'] = encode.fit_transform(crimeNormalData['INCIDENT_NUMBER'])
crimeNormalData['OCCURRED_ON_DATE'] = encode.fit_transform(crimeNormalData['OCCURRED_ON_DATE'])
crimeNormalData['DISTRICT'] = encode.fit_transform(crimeNormalData['DISTRICT'])
crimeNormalData['STREET'] = encode.fit_transform(crimeNormalData['STREET'])
# Print the first 5 rows to see the changes
print(crimeNormalData.head())
print(crimeNormalData['OFFENSE_DESCRIPTION'].head())
crimeNormalData.replace(' ', np.nan, inplace=True)

# Drop the rows where at least one element is missing.
crimeNormalData.dropna(inplace=True)
# now we will save the normalized data into the csv file so that it can be used for the further analysis.
# In the Machine Learning models.
newCrimeData = crimeNormalData.to_csv('crimeData_normal.csv', index=False)

# Load the cleaned and normalized data
crimeNormalData = pd.read_csv('crimeData_normal.csv')

# Split the data into features (X) and target (y)
# Assuming 'OFFENSE_CODE_GROUP' is the target and the rest are features
X = crimeNormalData.drop('OFFENSE_CODE_GROUP', axis=1)
y = crimeNormalData['OFFENSE_CODE_GROUP']


print(crimeNormalData['OCCURRED_ON_DATE'].head())


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Choose a machine learning model
model = KMeans(n_clusters=4, random_state=20)

# Train the model with the training data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model with the testing data
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")




# Choose a machine learning model for classification
clf = RandomForestClassifier(n_estimators=100, random_state=20)
# Train the model with the training data
clf.fit(X_train, y_train)
# Make predictions
predictions = clf.predict(X_test)
# Evaluate the model with the testing data
print(classification_report(y_test, predictions))
# Choose a machine learning model for clustering
model = KMeans(n_clusters=4, random_state=20)
# Train the model with the training data
model.fit(X_train)
# Make predictions
predictions = model.predict(X_test)
# Add the predictions to your dataframe
X_test['cluster'] = predictions
# Print the first few rows of your dataframe to see the clusters
print(X_test.head())

cm = confusion_matrix(y_test, predictions)

# Convert the confusion matrix to a dataframe
cm_df = pd.DataFrame(cm)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()






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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler

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


# Drop rows where 'Lat' and 'Long' are null
crimeNormalData = crimeNormalData.dropna(subset=['Lat', 'Long'])




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
# checking the shape of the csv file.
print("The shape of the data: ", crimeNormalData.shape)



# checking the null values of the data.
nullCount = crimeNormalData.isnull().sum()
print("The values of the csv that are null: \n", nullCount)


#
# # Create a label encoder
# encode = LabelEncoder()
# # Fit and transform the 'OFFENSE_CODE_GROUP' column
# crimeNormalData['OFFENSE_CODE_GROUP'] = encode.fit_transform(crimeNormalData['OFFENSE_CODE_GROUP'])
# crimeNormalData['OFFENSE_DESCRIPTION'] = encode.fit_transform(crimeNormalData['OFFENSE_DESCRIPTION'])
# crimeNormalData['INCIDENT_NUMBER'] = encode.fit_transform(crimeNormalData['INCIDENT_NUMBER'])
# crimeNormalData['OCCURRED_ON_DATE'] = encode.fit_transform(crimeNormalData['OCCURRED_ON_DATE'])
# crimeNormalData['DISTRICT'] = encode.fit_transform(crimeNormalData['DISTRICT'])
# crimeNormalData['STREET'] = encode.fit_transform(crimeNormalData['STREET'])
# # Print the first 5 rows to see the changes
# print(crimeNormalData.head())
# print(crimeNormalData['OFFENSE_DESCRIPTION'].head())
# crimeNormalData.replace(' ', np.nan, inplace=True)
#
# # Drop the rows where at least one element is missing.
# crimeNormalData.dropna(inplace=True)
# # now we will save the normalized data into the csv file so that it can be used for the further analysis.
# # In the Machine Learning models.
# newCrimeData = crimeNormalData.to_csv('crimeData_normal.csv', index=False)



#load the initial data
#crimeData = pd.read_csv('archive\crime.csv')

# load the cleaned data
crimeCleanData = pd.read_csv('crimeData_clean.csv')


# Load the cleaned and normalized data
crimeNormalData = pd.read_csv('crimeData_normal.csv')

# #label encoding
# encode = LabelEncoder()
# crimeCleanData['OFFENSE_CODE_GROUP'] = encode.fit_transform(crimeCleanData['OFFENSE_CODE_GROUP'])
# crimeCleanData['OFFENSE_DESCRIPTION'] = encode.fit_transform(crimeCleanData['OFFENSE_DESCRIPTION'])
# crimeCleanData['INCIDENT_NUMBER'] = encode.fit_transform(crimeCleanData['INCIDENT_NUMBER'])
# crimeCleanData['OCCURRED_ON_DATE'] = encode.fit_transform(crimeCleanData['OCCURRED_ON_DATE'])
# crimeCleanData['DISTRICT'] = encode.fit_transform(crimeCleanData['DISTRICT'])
# crimeCleanData['STREET'] = encode.fit_transform(crimeCleanData['STREET'])
#
# days_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
#
# # Use the map function to replace the days of the week with numerical values
# crimeCleanData['DAY_OF_WEEK'] = crimeCleanData['DAY_OF_WEEK'].map(days_mapping)
#
# # Create a dictionary to map UCR_PART to numerical values
# ucr_mapping = {'Part One': 1, 'Part Two': 2, 'Part Three': 3}
# # Use the map function to replace the UCR_PART with numerical values
# crimeCleanData['UCR_PART'] = crimeCleanData['UCR_PART'].map(ucr_mapping)
#
# print("The cleaned data: \n", crimeCleanData.head())
# # Standardize the data
# scaler = StandardScaler()
# crime_data_scaled = scaler.fit_transform(crimeCleanData)
# print(crime_data_scaled.head())
# # Now your data is ready for machine learning
#
#
#
#
# # Split the data into features (X) and target (y)
# # Assuming 'OFFENSE_CODE_GROUP' is the target and the rest are features
# X = crimeCleanData.drop('OFFENSE_CODE_GROUP', axis=1)
# y = crimeCleanData['OFFENSE_CODE_GROUP']
#
# print(crimeCleanData['OCCURRED_ON_DATE'].head())
#
# #feature scaling
# scaler = StandardScaler()
# crime_data_scaled = scaler.fit_transform(crimeCleanData)
#
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
#
# # Choose a machine learning model
# model = KMeans(n_clusters=4, random_state=20)
#
# # Train the model with the training data
# model.fit(X_train, y_train)
#
# # Make predictions
# predictions = model.predict(X_test)
#
# # Evaluate the model with the testing data
# accuracy = accuracy_score(y_test, predictions)
# print(f"Model Accuracy: {accuracy}")
#
# # Choose a machine learning model for classification
# clf = RandomForestClassifier(n_estimators=100, random_state=20)
# # Train the model with the training data
# clf.fit(X_train, y_train)
# # Make predictions
# predictions = clf.predict(X_test)
# # Evaluate the model with the testing data
# print(classification_report(y_test, predictions))
# # Choose a machine learning model for clustering
# model = KMeans(n_clusters=4, random_state=20)
# # Train the model with the training data
# model.fit(X_train)
# # Make predictions
# predictions = model.predict(X_test)
# # Add the predictions to your dataframe
# X_test['cluster'] = predictions
# # Print the first few rows of your dataframe to see the clusters
# print(X_test.head())
#
# cm = confusion_matrix(y_test, predictions)
#
# # Convert the confusion matrix to a dataframe
# cm_crimeCleanData = pd.DataFrame(cm)
#
# # Visualize the confusion matrix using a heatmap
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm_crimeCleanData, annot=True, fmt='g')
# plt.title('Confusion matrix of the classifier')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
#

print("unique district :",crimeCleanData['DISTRICT'].unique())
# # Define the parameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300, 400, 500],e
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }
#
# # Create a RandomForestClassifier
# clf = RandomForestClassifier()
#
# # Create a GridSearchCV object
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
#
# # Fit the GridSearchCV object to the data
# grid_search.fit(X_train, y_train)
#
# # Get the best parameters
# best_params = grid_search.best_params_
#
# # Create a RandomForestClassifier with the best parameters
# clf = RandomForestClassifier(**best_params)
#
# # Fit the model to the data
# clf.fit(X_train, y_train)
#
# # Make predictions
# predictions = clf.predict(X_test)
#
# # Evaluate the model
# print(classification_report(y_test, predictions))
#
# # Perform cross-validation
# scores = cross_val_score(clf, X, y, cv=5)
# print(f"Cross-Validation Accuracy Scores: {scores}")




import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Load the normalized data
normalized_data = pd.read_csv('crimeData_normalized.csv')

# Take a sample of 20,000
sample_data = normalized_data.sample(n=2000)

# Group the sample data by district
grouped_data = sample_data.groupby('DISTRICT')

# Create a scatter plot for each district
plt.figure(figsize=(10, 6))
for name, group in grouped_data:
    plt.scatter(group['Long'], group['Lat'], label=f'District {name}', alpha=0.5)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of Crime Locations by District')
plt.legend(title='District', markerscale=2)
plt.show()

# Perform clustering based on the district of crime
from sklearn.cluster import KMeans

# Extract the district column
districts = sample_data[['Lat', 'Long']]

# Create a KMeans object with 12 clusters
kmeans = KMeans(n_clusters=12)

# Fit the model to the data
kmeans.fit(districts)

# Get the cluster labels
labels = kmeans.labels_

# Plot the clusters
plt.figure(figsize=(10, 6))
for i, (name, group) in enumerate(grouped_data):
    plt.scatter(group['Long'], group['Lat'], label=f'District {name}', alpha=0.5, marker=f'${i}$')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], marker='x', s=100, c='red', label='Centroids')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clustered Crime Locations by District')
plt.legend(title='District', markerscale=2)
plt.show()

# Assign cluster labels to each data point
sample_data['Cluster'] = kmeans.labels_

# Group data by cluster
grouped_data = sample_data.groupby('Cluster')

# Initialize dictionaries to store the offense description with maximum count in each cluster
max_count_offense = {}
max_count = {}

# Loop through each cluster
for cluster, data in grouped_data:
    # Count occurrences of each offense description in the cluster
    offense_counts = data['OFFENSE_DESCRIPTION'].value_counts()
    # Get the offense description with maximum count
    max_count_offense[cluster] = offense_counts.idxmax()
    # Get the maximum count
    max_count[cluster] = offense_counts.max()

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar(max_count_offense.keys(), max_count.values())
plt.xlabel('Cluster')
plt.ylabel('Count of Maximum Crime')
plt.title('Count of Maximum Crime in Each Cluster')
plt.xticks(range(len(max_count_offense)), sorted(max_count_offense.keys()))  # Ensure correct order on x-axis
plt.legend(title='District', markerscale=2)
plt.show()


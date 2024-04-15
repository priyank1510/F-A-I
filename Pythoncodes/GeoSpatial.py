import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# # Load the crime data
# crime_data = pd.read_csv('crimeData_clean.csv')
#
# # Take a random sample of the data
# sample_size = 50000
# crime_data_sample = crime_data.sample(n=sample_size, random_state=42)
#
# # Data preprocessing
# crime_data_sample = crime_data_sample.dropna(subset=['Lat', 'Long'])  # Remove rows with missing latitude and longitude
# crime_data_sample = crime_data_sample[['Lat', 'Long']]  # Select relevant features
#
# # Standardize the data
# scaler = StandardScaler()
# crime_data_scaled = scaler.fit_transform(crime_data_sample)
#
# # DBSCAN Clustering
# dbscan = DBSCAN(eps=0.01, min_samples=10)  # Adjust eps and min_samples as needed
# dbscan_labels = dbscan.fit_predict(crime_data_scaled)
#
# # K-Means Clustering
# kmeans = KMeans(n_clusters=5)  # Adjust the number of clusters as needed
# kmeans_labels = kmeans.fit_predict(crime_data_scaled)
#
# # Function to create and save map
# def create_map_with_clusters(cluster_labels, method_name):
#     # Create a base map centered on Boston
#     base_map = folium.Map(location=[42.3601, -71.0589], zoom_start=12)  # Boston coordinates
#
#     # Add crime locations to the map
#     for idx, row in crime_data_sample.iterrows():
#         lat = row['Lat']
#         lon = row['Long']
#         folium.CircleMarker([lat, lon], radius=3, color='red').add_to(base_map)
#
#     # Add clusters to the map
#     cluster_colors = ['green', 'blue', 'purple', 'orange', 'darkred', 'cadetblue', 'lightgreen']
#     for label in set(cluster_labels):
#         if label != -1:
#             cluster_points = crime_data_sample[cluster_labels == label]
#             for idx, row in cluster_points.iterrows():
#                 lat = row['Lat']
#                 lon = row['Long']
#                 folium.CircleMarker([lat, lon], radius=3, color=cluster_colors[label % len(cluster_colors)]).add_to(base_map)
#
#     # Save the map to an HTML file
#     base_map.save(f'map_{method_name}.html')
#
# # Generate and save maps with clusters for DBSCAN and K-Means
# create_map_with_clusters(dbscan_labels, 'DBSCAN')
# create_map_with_clusters(kmeans_labels, 'KMeans')



#
#
#
#
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.cluster import KMeans
#
# # Load the normalized data
# normalized_data = pd.read_csv('crimeData_normalized.csv')
#
# # Take a sample of 20,000
# sample_data = normalized_data.sample(n=2000)
#
# # Group the sample data by district
# grouped_data = sample_data.groupby('DISTRICT')
#
# # Create a scatter plot for each district
# plt.figure(figsize=(10, 6))
# for name, group in grouped_data:
#     plt.scatter(group['Long'], group['Lat'], label=name, alpha=0.5)
#
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Scatter Plot of Crime Locations by District')
# plt.legend(title='District', markerscale=2)
# plt.show()
#
# # Perform clustering based on the district of crime
# from sklearn.cluster import KMeans
#
# # Extract the district column
# districts = sample_data[['Lat', 'Long']]
#
# # Create a KMeans object with 10 clusters
# kmeans = KMeans(n_clusters=12)
#
# # Fit the model to the data
# kmeans.fit(districts)
#
# # Get the cluster labels
# labels = kmeans.labels_
#
# # Plot the clusters
# plt.figure(figsize=(10, 6))
# for name, group in grouped_data:
#     plt.scatter(group['Long'], group['Lat'], label=name, alpha=0.5)
#
# # Plot the centroids
# plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], marker='x', s=100, c='red', label='Centroids')
#
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Clustered Crime Locations by District')
# plt.legend(title='District', markerscale=2)
# plt.show()
#

#
# import pandas as pd
# from sklearn.cluster import KMeans, DBSCAN
# import folium
# from folium.plugins import MarkerCluster
# from sklearn.preprocessing import StandardScaler
#
# # Load the crime data
# crime_data = pd.read_csv('crimeData_clean.csv')
#
# # Take a random sample of the data
# sample_size = 50000
# crime_data_sample = crime_data.sample(n=sample_size, random_state=42)
#
# # Data preprocessing
# crime_data_sample = crime_data_sample.dropna(subset=['Lat', 'Long'])  # Remove rows with missing latitude and longitude
# crime_data_sample = crime_data_sample[['Lat', 'Long']]  # Select relevant features
#
#
# district_names = ['D14', 'C11', 'D4', 'B3', 'B2', 'C6', 'A1', 'E5', 'A7', 'E13', 'E18', 'A15']
#
#
#
#
# # DBSCAN Clustering
# dbscan = DBSCAN(eps=0.01, min_samples=10)  # Adjust eps and min_samples as needed
# dbscan_labels = dbscan.fit_predict(crime_data_sample)
#
# # K-Means Clustering
# kmeans = KMeans(n_clusters=5)  # Adjust the number of clusters as needed
# kmeans_labels = kmeans.fit_predict(crime_data_sample)
#
# # Function to create and save map
# def create_map_with_clusters(cluster_labels, method_name):
#     # Create a base map centered on Boston
#     base_map = folium.Map(location=[42.3601, -71.0589], zoom_start=12)  # Boston coordinates
#
#     # Add crime locations to the map
#     for idx, row in crime_data_sample.iterrows():
#         lat = row['Lat']
#         lon = row['Long']
#         folium.CircleMarker([lat, lon], radius=3, color='red').add_to(base_map)
#
#     # Add clusters to the map
#     cluster_colors = ['green', 'blue', 'purple', 'orange', 'darkred', 'cadetblue', 'lightgreen']
#     for label in set(cluster_labels):
#         if label != -1:
#             cluster_points = crime_data_sample[cluster_labels == label]
#             for idx, row in cluster_points.iterrows():
#                 lat = row['Lat']
#                 lon = row['Long']
#                 folium.CircleMarker([lat, lon], radius=3, color=cluster_colors[label % len(cluster_colors)]).add_to(base_map)
#
#     # Save the map to an HTML file
#     base_map.save(f'map_{method_name}.html')
#
# # Generate and save maps with clusters for DBSCAN and K-Means
# create_map_with_clusters(dbscan_labels, 'DBSCAN')
# create_map_with_clusters(kmeans_labels, 'KMeans')
#
# # Load the normalized data
# normalized_data = pd.read_csv('crimeData_clean.csv')
#
# # Take a sample of 20,000
# sample_data = normalized_data.sample(n=100000)
#
# # Perform clustering based on the district of crime
# # Extract the latitude and longitude columns
# districts = sample_data[['Lat', 'Long']]
#
# # Create a KMeans object with 12 clusters
# kmeans = KMeans(n_clusters=12)
#
# # Fit the model to the data
# kmeans.fit(districts)
#
# # Add cluster labels to the sample data
# sample_data['Cluster'] = kmeans.labels_
#
# # Convert 'Cluster' column to integer
# sample_data['Cluster'] = sample_data['Cluster'].astype(int)
#
# # Create a map centered around Boston
# crime_map = folium.Map(location=[42.3601, -71.0589], zoom_start=12)
#
# # Create MarkerCluster for each cluster
# marker_clusters = [MarkerCluster().add_to(crime_map) for _ in range(12)]
#
# # Add crime locations to MarkerClusters
# for idx, row in sample_data.iterrows():
#     folium.Marker(
#         location=[row['Lat'], row['Long']],
#         popup=f"Cluster: {district_names[row['Cluster']]}, Offense Description: {row['OFFENSE_DESCRIPTION']}"
#     ).add_to(marker_clusters[row['Cluster']])
#
# # Display the map
# crime_map.save('crime_map.html')
#
#




# import pandas as pd
# from sklearn.cluster import KMeans
# import folium
# from folium.plugins import MarkerCluster
#
# # Load the crime data
# crime_data = pd.read_csv('crimeData_clean.csv')
#
# # Take a random sample of the data
# sample_size = 50000
# crime_data_sample = crime_data.sample(n=sample_size, random_state=42)
#
# # Data preprocessing
# crime_data_sample = crime_data_sample.dropna(subset=['Lat', 'Long'])  # Remove rows with missing latitude and longitude
# crime_data_sample = crime_data_sample[['Lat', 'Long', 'OCCURRED_ON_DATE', 'DISTRICT', 'OFFENSE_DESCRIPTION']]  # Select relevant features
#
# # Define district names
# district_names = ['D14', 'C11', 'D4', 'B3', 'B2', 'C6', 'A1', 'E5', 'A7', 'E13', 'E18', 'A15']
#
# # Extract hour from OCCURRED_ON_DATE column
# crime_data_sample['HOUR'] = pd.to_datetime(crime_data_sample['OCCURRED_ON_DATE']).dt.hour
#
# # Define time slots
# daytime_slots = range(7, 19)  # Daytime: 7 AM to 6:59 PM
# nighttime_slots = list(range(19, 24)) + list(range(0, 7))  # Nighttime: 7 PM to 6:59 AM
#
#
#
#
# # Split data into daytime and nighttime crimes
# daytime_crimes = crime_data_sample[crime_data_sample['HOUR'].isin(daytime_slots)]
# nighttime_crimes = crime_data_sample[crime_data_sample['HOUR'].isin(nighttime_slots)]
#
# # Calculate total number of daytime and nighttime crimes
# total_daytime_crimes = len(daytime_crimes)
# total_nighttime_crimes = len(nighttime_crimes)
#
#
# print("Total daytime crimes:", total_daytime_crimes)
# print("Total nighttime crimes:", total_nighttime_crimes)
#
#
#
# # Create a map centered around Boston for daytime crimes
# daytime_map = folium.Map(location=[42.3601, -71.0589], zoom_start=12)
#
# # Create MarkerCluster for each district
# daytime_marker_clusters = [MarkerCluster().add_to(daytime_map) for _ in range(12)]
#
# # Add daytime crime locations to MarkerClusters
# for idx, row in daytime_crimes.iterrows():
#     folium.Marker(
#         location=[row['Lat'], row['Long']],
#         popup=f"District: {row['DISTRICT']}, Offense Description: {row['OFFENSE_DESCRIPTION']}"
#     ).add_to(daytime_marker_clusters[district_names.index(row['DISTRICT'])])
#
# # Save the daytime map
# daytime_map.save('daytime_crime_map.html')
#
# # Create a map centered around Boston for nighttime crimes
# nighttime_map = folium.Map(location=[42.3601, -71.0589], zoom_start=12)
#
# # Create MarkerCluster for each district
# nighttime_marker_clusters = [MarkerCluster().add_to(nighttime_map) for _ in range(12)]
#
# # Add nighttime crime locations to MarkerClusters
# for idx, row in nighttime_crimes.iterrows():
#     folium.Marker(
#         location=[row['Lat'], row['Long']],
#         popup=f"District: {row['DISTRICT']}, Offense Description: {row['OFFENSE_DESCRIPTION']}"
#     ).add_to(nighttime_marker_clusters[district_names.index(row['DISTRICT'])])
#
# # Save the nighttime map
# nighttime_map.save('nighttime_crime_map.html')





import pandas as pd
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster

# Load the crime data
crime_data = pd.read_csv('crimeData_clean.csv')

# Data preprocessing
crime_data = crime_data.dropna(subset=['Lat', 'Long', 'DISTRICT'])  # Remove rows with missing values
crime_data['OCCURRED_ON_DATE'] = pd.to_datetime(crime_data['OCCURRED_ON_DATE'])  # Convert OCCURRED_ON_DATE to datetime

# Extract hour and district from the data
crime_data['HOUR'] = crime_data['OCCURRED_ON_DATE'].dt.hour
crime_data['DISTRICT'] = crime_data['DISTRICT'].astype(str)  # Convert DISTRICT to string

# Define time slots
daytime_slots = range(7, 19)  # Daytime: 7 AM to 6:59 PM
nighttime_slots = list(range(19, 24)) + list(range(0, 7))  # Nighttime: 7 PM to 6:59 AM

# Split data into daytime and nighttime crimes
daytime_crimes = crime_data[crime_data['HOUR'].isin(daytime_slots)]
nighttime_crimes = crime_data[crime_data['HOUR'].isin(nighttime_slots)]

# Perform K-Means clustering on daytime crimes
kmeans_daytime = KMeans(n_clusters=10, random_state=42)  # Adjust the number of clusters as needed
daytime_crimes['Cluster'] = kmeans_daytime.fit_predict(daytime_crimes[['Lat', 'Long']])

# Analyze daytime crime clusters
daytime_clusters = daytime_crimes.groupby('Cluster')
for cluster_id, cluster_data in daytime_clusters:
    print(f"\nCluster {cluster_id}:")
    print("Most common crime in this cluster:")
    print(cluster_data['OFFENSE_DESCRIPTION'].value_counts().head(1))
    print("Most common district for crimes in this cluster:")
    print(cluster_data['DISTRICT'].value_counts().head(1))

# Perform K-Means clustering on nighttime crimes
kmeans_nighttime = KMeans(n_clusters=12, random_state=42)  # Adjust the number of clusters as needed
nighttime_crimes['Cluster'] = kmeans_nighttime.fit_predict(nighttime_crimes[['Lat', 'Long']])

# Analyze nighttime crime clusters
nighttime_clusters = nighttime_crimes.groupby('Cluster')
for cluster_id, cluster_data in nighttime_clusters:
    print(f"\nCluster {cluster_id}:")
    print("Most common crime in this cluster:")
    top_crime = cluster_data['OFFENSE_DESCRIPTION'].value_counts().head(1)
    if not top_crime.empty:
        print(top_crime)
    else:
        print("No offense descriptions available for this cluster.")
    print("Most common district for crimes in this cluster:")
    print(cluster_data['DISTRICT'].value_counts().head(1))

# Analyze overall crime patterns
overall_crimes = crime_data.groupby(['OFFENSE_DESCRIPTION', 'DISTRICT'])['OCCURRED_ON_DATE'].count().reset_index().sort_values(['OCCURRED_ON_DATE'], ascending=False)
print("\nOverall most common crime types and districts:")
print(overall_crimes.head(10))

# Visualize daytime crime clusters on a map
daytime_map = folium.Map(location=[crime_data['Lat'].mean(), crime_data['Long'].mean()], zoom_start=12)
daytime_marker_clusters = [MarkerCluster().add_to(daytime_map) for _ in range(kmeans_daytime.n_clusters)]

for idx, row in daytime_crimes.iterrows():
    folium.Marker(
        location=[row['Lat'], row['Long']],
        popup=f"Cluster: {row['Cluster']}, District: {row['DISTRICT']}, Offense Description: {row['OFFENSE_DESCRIPTION']}"
    ).add_to(daytime_marker_clusters[row['Cluster']])

daytime_map.save('daytime_crime_clusters.html')

# Visualize nighttime crime clusters on a map
nighttime_map = folium.Map(location=[crime_data['Lat'].mean(), crime_data['Long'].mean()], zoom_start=12)
nighttime_marker_clusters = [MarkerCluster().add_to(nighttime_map) for _ in range(kmeans_nighttime.n_clusters)]

for idx, row in nighttime_crimes.iterrows():
    folium.Marker(
        location=[row['Lat'], row['Long']],
        popup=f"Cluster: {row['Cluster']}, District: {row['DISTRICT']}, Offense Description: {row['OFFENSE_DESCRIPTION']}"
    ).add_to(nighttime_marker_clusters[row['Cluster']])

nighttime_map.save('nighttime_crime_clusters.html')

# Provide recommendations based on the analysis
print("\nRecommendations:")
print("Based on the analysis, the following areas or districts might be advisable to avoid:")
high_risk_districts = overall_crimes['DISTRICT'].value_counts().head(3).index.tolist()
print(f"- {', '.join(high_risk_districts)} (due to overall high crime rates)")

high_risk_nighttime_clusters = nighttime_clusters.apply(lambda x: x['OFFENSE_DESCRIPTION'].value_counts().index[0] if not x['OFFENSE_DESCRIPTION'].value_counts().empty else '').value_counts().head(2).index.tolist()
high_risk_crimes = [crime for crime in high_risk_nighttime_clusters if crime != '']
print(f"- Areas around nighttime crime clusters {', '.join(map(str, high_risk_nighttime_clusters))} (due to high rates of {', '.join(high_risk_crimes)})")






# #the heatmap of the crime data.
import folium
from folium.plugins import HeatMap
# Load the crime data
crime_data = pd.read_csv(r'C:\Users\patel\Documents\CS-5100 Foundation To Artificial Intelligence\Project-work\Pythoncodes\crimeData_clean.csv')

# # Create a base map
# heat_crime_map = folium.Map(location=[crime_data['Lat'].mean(), crime_data['Long'].mean()], zoom_start=12)
#
# # Add crime locations to the map
# crime_locations = crime_data[['Lat', 'Long']].values.tolist()
#
# # Create a heatmap layer
# heatmap_layer = HeatMap(crime_locations, radius=15)
#
# # Add the heatmap layer to the map
# heatmap_layer.add_to(heat_crime_map)
#
# # Display the map
# heat_crime_map
# heat_crime_map.save('heat_crime_map.html')
#
#

# Count the occurrences of each crime type
crime_type_counts = crime_data['OFFENSE_DESCRIPTION'].value_counts()

# Print the top 10 most common crime types
print("Top 10 Most Common Crime Types:")
print(crime_type_counts.head(10))

# Analyze crime types within a specific district
district_of_interest = 'D14'
district_crimes = crime_data[crime_data['DISTRICT'] == district_of_interest]
district_crime_types = district_crimes['OFFENSE_DESCRIPTION'].value_counts()

print(f"\nTop Crime Types in District {district_of_interest}:")
print(district_crime_types.head(10))
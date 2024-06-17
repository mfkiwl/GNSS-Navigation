import pandas as pd
from geopy.distance import geodesic
import numpy as np

# Load the data
df1 = pd.read_csv('position_data.csv')
df2 = pd.read_csv('data/2020-08-13-21-42-us-ca-mtv-sf-280/pixel5/ground_truth.csv')

# Convert utcTimeMillis to datetime
df1['UnixTimeMillis'] = pd.to_datetime(df1['UnixTimeMillis'], unit='ms')
df2['UnixTimeMillis'] = pd.to_datetime(df2['UnixTimeMillis'], unit='ms')

# Merge the dataframes on the utcTimeMillis column
merged_df = pd.merge(df1, df2, on='UnixTimeMillis', suffixes=('_file1', '_file2'))

# Calculate the distance between coordinates
def calculate_distance(row):
    coords_1 = (row['LatitudeDegrees_file1'], row['LongitudeDegrees_file1'])
    coords_2 = (row['LatitudeDegrees_file2'], row['LongitudeDegrees_file2'])
    return geodesic(coords_1, coords_2).meters

merged_df['distance'] = merged_df.apply(calculate_distance, axis=1)

distance = merged_df['distance'].to_numpy()
score = np.mean([np.quantile(distance, 0.50), np.quantile(distance, 0.95)])

print(f'The average distance between the coordinates is {score:.4f} meters.')

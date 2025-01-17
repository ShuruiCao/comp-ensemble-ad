import pandas as pd
import numpy as np
from geopy.distance import geodesic
from scipy.interpolate import interp1d
import random
import math
# Function to calculate the Haversine distance between two geographical points
def haversine_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters
EARTH_RADIUS_M = 6371000.0
from geopy.distance import geodesic
def get_distance_between_points_haversine(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
):
    """
    Returns the distance between two points on the Earth's surface, given
    their latitudes and longitudes. Uses an approximation formula
    that is 2x faster than haversine and is accurate up to 10,000 m.
    """

    lat1, lon1, lat2, lon2 = (
        np.radians(lat1),
        np.radians(lon1),
        np.radians(lat2),
        np.radians(lon2),
    )
    x = (lon2 - lon1) * np.cos((lat1 + lat2) / 2)
    y = lat2 - lat1
    d = np.sqrt(x**2 + y**2) * EARTH_RADIUS_M
    return d
# Function to interpolate points with consistent distances based on old and new distances
def interpolate_to_maintain_distance(start_point, end_point, old_dist, new_dist, segment1):
    """
    Interpolates points between two given points to maintain consistent distance, derived from the original distance (old_dist).
    
    Parameters:
    - start_point (DataFrame row): The starting point for interpolation.
    - end_point (DataFrame row): The ending point for interpolation.
    - old_dist (float): The original distance before adding offset.
    - new_dist (float): The new distance after adding offset.
    
    Returns:
    - DataFrame: Interpolated points to maintain consistent distance.
    """
    # Calculate the number of interpolated points needed
    # num_interpolations = int(new_dist / old_dist)  # Number of interpolated points
    num_interpolations = math.ceil(new_dist / old_dist) 
    NUMtoINT = num_interpolations + 6
    # if segment1 == True:
    #     NUMtoINT = num_interpolations + 3
    # elif segment1 == False:
    #     NUMtoINT = num_interpolations + 5
    
    # Interpolation based on the number of points needed
    t = np.linspace(0, 1, NUMtoINT)  # Including start and end points and one more for backup
    lat_interp = interp1d([0, 1], [start_point['latitude'], end_point['latitude']], kind='linear')
    long_interp = interp1d([0, 1], [start_point['longitude'], end_point['longitude']], kind='linear')

    interpolated_points = pd.DataFrame({
        'latitude': lat_interp(t),
        'longitude': long_interp(t)
    })

    return interpolated_points

# Function to generate detour anomalies with consistent distance interpolation
def generate_detour_anomalies(trip_values, trip_keys, percentage_points_to_modify, num_anomalies, synthetic_generated_keys_detour, map_bbox):
    """
    Generates synthetic detour anomalies by modifying a percentage of points, using interpolation to maintain consistent distance.
    
    Parameters:
    - trip_values (list of DataFrames): List of real trips.
    - trip_keys (list): List of keys for the real trips.
    - percentage_points_to_modify (float): Percentage of points to modify (between 0 and 1).
    - num_anomalies (int): Number of anomalies to generate.
    
    Returns:
    - dict: Dictionary of synthetic anomalous trips with detours.
    """
    anomalies_detour = {}
    half_anomalies = num_anomalies // 2

    trip_count = 0
    max_attempts = 5000  # Maximum number of attempts to find valid trips
    attempts = 0
    lat_max, lat_min, lon_max, lon_min = map_bbox
    clipped_count = 0

    while trip_count < num_anomalies and attempts < max_attempts:
        # Randomly select a trip to modify
        trip_index = random.randint(0, len(trip_values) - 1)
        normal_trip = trip_values[trip_index].copy()
        normal_trip.reset_index(drop=True)

        # record speed of normal trip
        speed_normal = get_distance_between_points_haversine(
                normal_trip['latitude'], normal_trip['longitude'], 
                normal_trip['latitude'].shift(), normal_trip['longitude'].shift()
            )
        speed_normal = np.nan_to_num(speed_normal, nan=0)
        max_speed_normal = max(speed_normal)

        # Determine the total number of points and the portion to modify
        total_points = len(normal_trip)
        num_points_to_modify = int(total_points * percentage_points_to_modify)

        # Compute standard deviations for latitude and longitude
        lat_std_dev = normal_trip['latitude'].std()
        long_std_dev = normal_trip['longitude'].std()

        # Identify the start and end indices for the middle section (detour segment)
        start_index = (total_points - num_points_to_modify) // 2
        end_index = start_index + num_points_to_modify

        # Determine old_dist and new_dist for both segments
        segment_1_end = normal_trip.iloc[start_index - 1]  # Last point before the detour
        detour_start = normal_trip.iloc[start_index]       # First point of the detour

        # Old distance between Segment 1 and Detour Segment
        old_dist_1 = haversine_distance(
            segment_1_end['latitude'], segment_1_end['longitude'],
            detour_start['latitude'], detour_start['longitude']
        )
        # Determine old and new distances for Detour Segment and Segment 3
        detour_end = normal_trip.iloc[end_index - 1]
        segment_3_start = normal_trip.iloc[end_index]
        
        old_dist_2 = haversine_distance(
            detour_end['latitude'], detour_end['longitude'],
            segment_3_start['latitude'], 
            segment_3_start['longitude']
        )

        # if i < half_anomalies:
        #     # Apply latitude offset to the Detour Segment
        #     for idx in range(start_index, end_index):
        #         normal_trip.loc[idx, 'latitude'] += 2 * lat_std_dev

        # else:
        #     # Apply longitude offset for the Detour Segment
        #     for idx in range(start_index, end_index):
        #         normal_trip.loc[idx, 'longitude'] += 2 * long_std_dev

        for idx in range(start_index, end_index):
                normal_trip.loc[idx, 'latitude'] += 2 * lat_std_dev


        # Calculate new distance between Segment 1 and Detour Segment after offset
        new_dist_1 = haversine_distance(
            segment_1_end['latitude'], segment_1_end['longitude'],
            normal_trip.loc[start_index, 'latitude'], 
            normal_trip.loc[start_index, 'longitude']
        )

        # Interpolation to maintain consistent distance for Segment 1 and Detour Segment
        interpolated_segment_1 = interpolate_to_maintain_distance(segment_1_end, 
                                                                  normal_trip.loc[start_index], 
                                                                  old_dist_1, 
                                                                  new_dist_1,True)
        

        # Calculate new distance after offset for Detour Segment and Segment 3
        new_dist_2 = haversine_distance(
            detour_end['latitude'], 
            detour_end['longitude'], 
            normal_trip.loc[end_index, 'latitude'], 
            normal_trip.loc[end_index, 'longitude']
        )

        # Interpolate to maintain consistent distance for Detour Segment and Segment 3
        interpolated_segment_3 = interpolate_to_maintain_distance(detour_end, 
                                                                  normal_trip.loc[end_index], 
                                                                  old_dist_2, 
                                                                  new_dist_2,False)
        len_segment_head = len(normal_trip[:start_index])
        len_segment_tail = len(normal_trip[end_index:])
        # Reconstruct the trip with interpolated points to maintain consistent distance
        normal_trip = pd.concat([normal_trip[:start_index], interpolated_segment_1, 
                                 normal_trip[start_index:end_index], 
                                 interpolated_segment_3, 
                                 normal_trip[end_index:]])
        
        # Speed Check
        speed_syn = get_distance_between_points_haversine(
                normal_trip['latitude'], normal_trip['longitude'], 
                normal_trip['latitude'].shift(), normal_trip['longitude'].shift()
            )
        speed_syn = np.nan_to_num(speed_syn, nan=0)
        max_speed_syn = max(speed_syn)
        threshold = max_speed_normal + 5

        if max_speed_syn <= threshold:  # 10 m/s allowed increase
            start_time = normal_trip['timestamp_dt'].iloc[0]
            time_interval = pd.to_timedelta('5s')  # Assuming each row is 5 seconds apart in the original trip
            adjusted_times = [start_time + i * time_interval for i in range(len(normal_trip))]
            normal_trip.loc[:, 'timestamp'] = adjusted_times
            # Add the label column
            normal_trip.loc[:, 'label'] = 0  # Default label for non-detour segments
            normal_trip.loc[len_segment_head:len(normal_trip) - len_segment_tail - 1, 'label'] = 1

            # Ensure latitudes and longitudes are within the bounding box
            if ((normal_trip['latitude'] > lat_max).any() or 
                (normal_trip['latitude'] < lat_min).any() or 
                (normal_trip['longitude'] > lon_max).any() or 
                (normal_trip['longitude'] < lon_min).any()):
                print(f"Adjusting lat/lon for trip key: {tripkey}")
                normal_trip['latitude'] = normal_trip['latitude'].clip(lower=lat_min, upper=lat_max)
                normal_trip['longitude'] = normal_trip['longitude'].clip(lower=lon_min, upper=lon_max)
                clipped_count += 1
            anomalous_trip = normal_trip[['latitude', 'longitude','timestamp','label']]

            tripkey = trip_keys[trip_index]
            synthetic_generated_keys_detour[trip_index] = tripkey
            key = f"Detour_{tripkey}_{percentage_points_to_modify}"
            anomalies_detour[key] = anomalous_trip

            trip_count += 1

        attempts += 1
    print(f'Number of trips with clipped lat/lon: {clipped_count}')
    print(f'Number of trips: {len(anomalies_detour)}')
    return anomalies_detour



### old ver

# def generate_detour_anomalies(trip_values, trip_keys, percentage_points_to_modify, num_anomalies, synthetic_generated_keys_detour):
#     """
    # Generates synthetic detour anomalies by modifying a percentage of points.

    # Parameters:
    # - trip_values (list of DataFrames): List of real trips.
    # - trip_keys (list): List of keys for the real trips.
    # - percentage_points_to_modify (float): Percentage of points to modify (between 0 and 1).
    # - num_anomalies (int): Number of anomalies to generate.
    
    # Returns:
    # - dict: Dictionary of synthetic anomalous trips with detours.
    # """
    # anomalies_detour = {}
    # half_anomalies = num_anomalies // 2

    # for i in range(num_anomalies):
    #     # Randomly select a trip to modify
    #     trip_index = random.randint(0, len(trip_values) - 1)
    #     normal_trip = trip_values[trip_index].copy()
    #     normal_trip = normal_trip.reset_index(drop=True)

    #     # Compute standard deviations for latitude and longitude
    #     lat_std_dev = normal_trip['latitude'].std()
    #     long_std_dev = normal_trip['longitude'].std()
    #     # print(lat_std_dev,long_std_dev)

    #     # Determine the total number of points and the portion to modify
    #     total_points = len(normal_trip)
    #     num_points_to_modify = int(total_points * percentage_points_to_modify)

    #     # Identify the start and end indices for the middle section
    #     start_index = (total_points - num_points_to_modify) // 2
    #     end_index = start_index + num_points_to_modify
        
    #     # Apply offset to latitude for the first half and longitude for the second half
    #     if i < half_anomalies:
    #         # Apply latitude offset for the first half of anomalies
    #         for idx in range(start_index, end_index):
    #             normal_trip.loc[idx, 'latitude'] += 2*lat_std_dev
    #     else:
    #         # Apply longitude offset for the second half of anomalies
    #         for idx in range(start_index, end_index):
    #             normal_trip.loc[idx, 'longitude'] += 2*long_std_dev

    #     # Save the anomalous trip
    #     tripkey = trip_keys[trip_index]
    #     synthetic_generated_keys_detour[trip_index] = tripkey
    #     key = f"Detour_{tripkey}_{percentage_points_to_modify}"
    #     anomalies_detour[key] = normal_trip[['latitude', 'longitude']]

    # return anomalies_detour



def rotate_geographic_segment(segment, angle, origin):
    # Using 2D rotation formulas here
    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Rotation
    rotated_segment = []
    for lat, lon in zip(segment['latitude'], segment['longitude']):
        # Convert to Cartesian coordinates for rotation
        x, y = lat - origin[0], lon - origin[1]
        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)

        # Convert back to geographic coordinates
        lat_rot = x_rot + origin[0]
        lon_rot = y_rot + origin[1]
        rotated_segment.append((lat_rot, lon_rot))

    return pd.DataFrame(rotated_segment, columns=['latitude', 'longitude'])

def generate_triangular_detour_trip(normal_trip, segment_start_ratio, segment_end_ratio, rotation_angle):
    # Select segment C to D
    segment_start_index = int(len(normal_trip) * segment_start_ratio)
    segment_end_index = int(len(normal_trip) * segment_end_ratio)
    segment_cd = normal_trip.iloc[segment_start_index:segment_end_index]

    # Rotate segment around C
    origin_c = (segment_cd.iloc[0]['latitude'], segment_cd.iloc[0]['longitude'])
    rotated_around_c = rotate_geographic_segment(segment_cd, rotation_angle, origin_c)

    # Rotate segment around D
    origin_d = (segment_cd.iloc[-1]['latitude'], segment_cd.iloc[-1]['longitude'])
    rotated_around_d = rotate_geographic_segment(segment_cd, -rotation_angle, origin_d)

    # Concatenate rotated segments with the rest of the trip
    trip_before_segment = normal_trip.iloc[:segment_start_index]
    trip_after_segment = normal_trip.iloc[segment_end_index:]
    detoured_trip = pd.concat([trip_before_segment, rotated_around_c, rotated_around_d, trip_after_segment]).reset_index(drop=True)

    return detoured_trip

def add_detour(chosen_trip,alpha):
    lowerbound=0.5 - alpha/2
    upperbound=0.5 + alpha/2
    rotation_angle=50
    normal_trip = chosen_trip.copy()
    normal_trip['timestamp'] = range(len(normal_trip))

    synthetic_trip = generate_triangular_detour_trip(normal_trip, lowerbound,upperbound,rotation_angle)
    
    # start_time = pd.to_datetime(normal_trip.iloc[0]['timestamp'])
    # time_interval = pd.to_timedelta('1s')  # Assuming each row is 1 second apart in the original trip
    # start_time = normal_trip.iloc[0]['timestamp']
    # adjusted_times = [start_time + i * 1 for i in range(len(synthetic_trip))]
    # synthetic_trip = synthetic_trip.reset_index(drop=True)
    # synthetic_trip.loc[:, 'timestamp'] = adjusted_times

    synthetic_trip = synthetic_trip[['latitude','longitude']]

    return synthetic_trip

# def generate_detour_anomalies(trip_values, trip_keys, alpha, num_anomalies, synthetic_generated_keys_detour):
#     """
#     alpha controls the percent of detour added to the trip

#     Parameters:
#     trip_dict (dict): Dictionary of the real trips.
#     num_anomalies (int): Number of anomalies to generate.

#     Returns:
#     dict: Dictionary of synthetic anomalous trips.
#     """
#     anomalies_detour = {}
#     for i in range(num_anomalies):
#         # Randomly choose a normal trip to modify
#         trip_index = random.randint(0, len(trip_values) - 1)
#         normal_trip = trip_values[trip_index].copy()
#         normal_trip = normal_trip.reset_index(drop=True)

#         # Create an anomalous trip 
#         anomalous_trip = add_detour(normal_trip,alpha)
        
#         # Save the anomalous trip
#         tripkey = trip_keys[trip_index]
#         synthetic_generated_keys_detour[trip_index] = tripkey
#         key = f"Detour_{tripkey}_{alpha}"
#         anomalies_detour[key] = anomalous_trip

#     return anomalies_detour
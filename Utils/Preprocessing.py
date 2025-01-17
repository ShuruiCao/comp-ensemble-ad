import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

EARTH_RADIUS_M = 6371000.0

class TripProcessor:
    def __init__(self, dataset, data_directory, output_directory):
        self.dataset = dataset
        self.data_directory = data_directory
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the Haversine distance."""
        lat1, lon1, lat2, lon2 = (
            np.radians(lat1),
            np.radians(lon1),
            np.radians(lat2),
            np.radians(lon2),
        )
        x = (lon2 - lon1) * np.cos((lat1 + lat2) / 2)
        y = lat2 - lat1
        return np.sqrt(x**2 + y**2) * EARTH_RADIUS_M

    @staticmethod
    def cartesian_distance(lat1, lon1, lat2, lon2):
        """Calculate the Cartesian distance."""
        return np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

    @staticmethod
    def save_to_pickle(data, file_path):
        """Save data to a pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def resample_trip(self, df, new_time_step):
        """Resample a trip to a uniform time step."""
        total_duration_nor = df['timestamp_nor'].iloc[-1]
        if total_duration_nor < new_time_step or len(df) <= 1:
            return pd.DataFrame({
                'timestamp_nor': [0, new_time_step],
                'delta_x_nor_re': [0, df['delta_x_nor'].iloc[-1]],
                'delta_y_nor_re': [0, df['delta_y_nor'].iloc[-1]]
            })

        new_timestamps_nor = np.arange(0, total_duration_nor, new_time_step)
        return pd.DataFrame({
            'timestamp_nor': new_timestamps_nor,
            'delta_x_nor_re': np.interp(new_timestamps_nor, df['timestamp_nor'], df['delta_x_nor']),
            'delta_y_nor_re': np.interp(new_timestamps_nor, df['timestamp_nor'], df['delta_y_nor']),
        })

    def interpolate_trip(self, trip, step_size=50.0):
        """Interpolate a trip based on the cumulative distance."""
        max_dist = trip['cumulative_dist'].max()
        desired_dists = np.arange(0, max_dist, step_size)
        interp_x = np.interp(desired_dists, trip['cumulative_dist'], trip['latitude'])
        interp_y = np.interp(desired_dists, trip['cumulative_dist'], trip['longitude'])
        return pd.DataFrame({'cumulative_dist': desired_dists, 'latitude': interp_x, 'longitude': interp_y})

    def compute_route_speeds_and_shapes(self, trip_dict):
        """Compute route speeds and shapes for a set of trips."""
        routespeed = {}
        shape = {}

        for key, trip_df in tqdm(trip_dict.items(), desc="Processing Trips"):
            trip_output = trip_df.copy()

            # Compute route-related features
            start_lat, start_lon = trip_output['latitude'].iloc[0], trip_output['longitude'].iloc[0]
            trip_output['latitude'] -= start_lat
            trip_output['longitude'] -= start_lon
            trip_output['displacement_m_real'] = self.haversine_distance(
                trip_output['latitude'], trip_output['longitude'],
                trip_output['latitude'].shift(), trip_output['longitude'].shift()
            ).fillna(0)
            trip_output['cumulative_dist'] = trip_output['displacement_m_real'].cumsum()
            trip_output['speed'] = trip_output['displacement_m_real']
            routespeed[key] = trip_output[['latitude', 'longitude', 'cumulative_dist', 'speed']]

            # Compute shape-related features
            trip_output = trip_output[(trip_output['latitude'] != trip_output['latitude'].shift()) |
                                      (trip_output['longitude'] != trip_output['longitude'].shift())]
            trip_output['displacement_m'] = self.cartesian_distance(
                trip_output['latitude'], trip_output['longitude'],
                trip_output['latitude'].shift(), trip_output['longitude'].shift()
            ).fillna(0)
            S = trip_output['displacement_m'].cumsum().iloc[-1]
            trip_output['delta_x_nor'] = trip_output['latitude'] / S
            trip_output['delta_y_nor'] = trip_output['longitude'] / S
            trip_output['timestamp'] = range(len(trip_output))
            trip_output['timestamp_nor'] = trip_output['timestamp'] / S
            resampled_data = self.resample_trip(trip_output, new_time_step=24)
            delta_scaled = self.scaler.fit_transform(resampled_data[['delta_x_nor_re', 'delta_y_nor_re']])
            delta_pca = self.pca.fit_transform(delta_scaled)
            resampled_data['delta_x'], resampled_data['delta_y'] = delta_pca[:, 0], delta_pca[:, 1]
            shape[key] = resampled_data[['delta_x', 'delta_y']]

        return routespeed, shape

    def apply_interpolation_and_save(self, dataset, dataset_name):
        """Interpolate and save trips."""
        interpolated_dataset = {key: self.interpolate_trip(trip) for key, trip in dataset.items()}
        save_path = os.path.join(self.output_directory, f'route_{dataset_name}.pkl')
        self.save_to_pickle(interpolated_dataset, save_path)

    def process(self, dicts_to_process):
        """Process a set of trip dictionaries."""
        for dict_name, trip_dict in dicts_to_process.items():
            routespeed, shape = self.compute_route_speeds_and_shapes(trip_dict)
            self.save_to_pickle(routespeed, os.path.join(self.output_directory, f'routespeed_{dict_name}_evaluation.pkl'))
            self.save_to_pickle(shape, os.path.join(self.output_directory, f'shape_{dict_name}_evaluation.pkl'))
            self.apply_interpolation_and_save(routespeed, dict_name)
        print("Data processing and saving completed.")


# Example usage
if __name__ == "__main__":
    dataset_name = 'porto'
    data_directory = f'/data/shurui/ICDE2020_GMVSAE/data/injected_outliers/{dataset_name}'
    output_directory = f'/data/shurui/SyntheticEval/{dataset_name}/Processed'

    detour = pd.read_pickle(f'{data_directory}/detour_evaluation_gps.pkl')
    switch = pd.read_pickle(f'{data_directory}/switching_evaluation_gps.pkl')
    dicts_to_process = {'detour': detour, 'switching': switch}

    processor = TripProcessor(dataset_name, data_directory, output_directory)
    processor.process(dicts_to_process)

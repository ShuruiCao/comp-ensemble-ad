import random
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tqdm import tqdm

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def normalize_sequence(sequence, global_mean, global_std):
#     # Ensure global_std is not zero to avoid division by zero
#     # global_std[global_std == 0] = 1
#     # Normalize the sequence with global mean and standard deviation
#     normalized_sequence = (sequence - global_mean) / global_std
#     return normalized_sequence

class CustomDataset(Dataset):
    def __init__(self, data_dict, columns, datatype, normalize,datasetname):
        self.data = []
        self.normalize = normalize
        self.datatype = datatype
        self.datasetname = datasetname

        for _, trip in data_dict.items():

            sequence = trip[columns].values
            
            if self.normalize:
                if self.datatype == 'route':
                    if self.datasetname == 'porto':
                        mean_lat = 0.00335
                        mean_long = 0.00289
                        std_lat = 0.13490
                        std_long = 0.11275
                    elif self.datasetname == 'LA':
                        mean_lat = 0.00010
                        mean_long = 0.00137
                        std_lat = 0.02230
                        std_long = 0.03101
                    elif self.datasetname == 'Chengdu':
                        mean_lat = 0.00038854
                        mean_long = -0.0008040
                        std_lat = 0.01157721
                        std_long = 0.01412195
                    sequence = (sequence - np.array([mean_lat, mean_long])) / np.array([std_lat, std_long])
                elif self.datatype == 'speed':
                    if self.datasetname == 'porto':
                        mean = 132.90087401711253
                        std = 364.997953468235
                    elif self.datasetname == 'LA':
                        mean = 77.92937
                        std = 45.07313
                    elif self.datasetname == 'Chengdu':
                        mean = 18.2319
                        std = 18.301
                    sequence = (sequence - mean) / std
                

            self.data.append(torch.tensor(sequence, dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(batch):
        # Sort batch by decreasing lengths to allow packed padded sequences
        # batch.sort(key=lambda x: len(x), reverse=True)
        return pad_sequence(batch, batch_first=True, padding_value=0.0)


def load_data_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def filter_trips(data_dict, lower_threshold, upper_threshold):
    """
    Filters out trips from the data dictionary 
    """
    return {key: value for key, value in data_dict.items() if len(value) >= lower_threshold and len(value) <= upper_threshold}


def create_data_loaders(config,data_dict):
    # Load the data from the pickle file
    # data_dict = load_data_from_pickle(config['data_path'])

    if config['dataset_type'] == 'speed':
        columns = ['speed']
    elif config['dataset_type'] == 'route':
        columns = ['latitude', 'longitude']
    elif config['dataset_type'] == 'shape':
        columns = ['delta_x', 'delta_y']

    # Create dataset from the loaded data
    need_to_normalize = config.get('normalize', False)
    datatype = config['dataset_type']
    datasetname = config['dataset']
    filtered_data_dict = filter_trips(data_dict, 3, 150000) 
    dataset = CustomDataset(filtered_data_dict, columns, datatype, normalize=need_to_normalize, datasetname=datasetname)
    
    loader = DataLoader(dataset, shuffle=False, batch_size=config['batch_size'], drop_last=True,collate_fn=CustomDataset.collate_fn,num_workers=5)

    return loader


def encode_data_dict(trainer, trip_dict):
    """Encode sequences from the data dictionary."""
    embeddings_dict = {}
    trainer.model.eval()
    with torch.no_grad():
        for key, trip_data in tqdm(trip_dict.items(), desc="Encoding Trips", dynamic_ncols=True):
            specific_sequence = np.array([trip_data[col].values for col in ['delta_x', 'delta_y']]).T
            trip_tensor = torch.tensor(specific_sequence, dtype=torch.float32).unsqueeze(0).to(trainer.device) # unsqueeze to add batch dimension
            encoded_representation = trainer.model.encode_sequence(trip_tensor)
            embeddings_dict[key] = encoded_representation
    return embeddings_dict

import random
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
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
                    sequence = (sequence - np.array([mean_lat, mean_long])) / np.array([std_lat, std_long])
                elif self.datatype == 'speed':
                    if self.datasetname == 'porto':
                        mean = 132.90087401711253
                        std = 364.997953468235
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

    if config['data_type'] == 'speed':
        columns = ['speed']
    elif config['data_type'] == 'route':
        columns = ['latitude', 'longitude']
    elif config['data_type'] == 'shape':
        columns = ['delta_x', 'delta_y']

    # Create dataset from the loaded data
    need_to_normalize = config.get('normalize', False)
    datatype = config['data_type']
    datasetname = config['dataset']
    filtered_data_dict = filter_trips(data_dict, 10, 1000)
    dataset = CustomDataset(filtered_data_dict, columns, datatype, normalize=need_to_normalize, datasetname=datasetname)
    
    loader = DataLoader(dataset, shuffle=False, batch_size=config['batch_size'], drop_last=True,collate_fn=CustomDataset.collate_fn,num_workers=5)

    return loader

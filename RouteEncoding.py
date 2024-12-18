# Imports
import torch
import torch.optim as optim
from Data_utils import set_seed, create_data_loaders, encode_data_dict, filter_trips
from LSTM_model import LSTMAutoencoder, LSTMAutoencoderTrainer
import os
import pandas as pd
import numpy as np
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle

#### FUNCTIONS
# ROUTE

dataset='Chengdu'
# Porto
if dataset == 'Porto':
    normalization_params = {
        'mean_lat': 0.00335,
        'mean_long': 0.00288,
        'std_lat': 0.13489,
        'std_long': 0.11274
    }
    datasetsaving = 'porto'
elif dataset == 'LA':
    normalization_params = {
       'mean_lat': 0.00010,
        'mean_long': 0.00137,
        'std_lat': 0.02230,
        'std_long': 0.03101
    }
    datasetsaving = 'LA'
elif dataset == 'Chengdu':
    normalization_params = {
        'mean_lat': 0.00038854,
        'mean_long': -0.0008040,
        'std_lat': 0.01157721,
        'std_long':  0.01412195
    }
    datasetsaving = 'Chengdu'

def normalize_sequence(sequence, normalization_params):
    mean_values = np.array([normalization_params['mean_lat'], normalization_params['mean_long']])
    std_values = np.array([normalization_params['std_lat'], normalization_params['std_long']])
    # Apply normalization
    normalized_sequence = (sequence - mean_values) / std_values
    return normalized_sequence

def normalizedict(datadict):
    # Relevant columns to extract
    relevant_columns = ['latitude', 'longitude']
    # Dictionary to hold normalized sequences
    normalized_sequences = {}
    # Extract relevant columns and normalize sequences
    for key, dataframe in datadict.items():
        # Extract relevant columns to create a numpy array
        sequence = np.array([dataframe[col].values for col in relevant_columns]).T
        # Normalize the sequence
        normalized_sequence = normalize_sequence(sequence, normalization_params)
        # Store the normalized sequence in the dictionary
        normalized_sequences[key] = normalized_sequence
    return normalized_sequences

def custom_collate_fn(batch):
    # Find the longest sequence in the batch
    max_len = max([seq.shape[0] for seq in batch])
    # Pad each sequence to the maximum length
    padded_sequences = [np.pad(seq, ((0, max_len - seq.shape[0]), (0, 0)), mode='constant') for seq in batch]
    # Convert to a tensor
    return torch.tensor(padded_sequences, dtype=torch.float32)

# Custom dataset for normalized sequences
class NormalizedSequenceDataset(Dataset):
    def __init__(self, data):
        self.data = list(data.values())  # Store sequences as list of arrays
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# Function to apply autoencoder, pad sequences, and calculate embeddings and MSE loss
def apply_autoencoder_and_calculate_loss(autoencoder, normalized_sequences, device, batch_size=64):
    """Apply autoencoder on normalized sequences and calculate MSE loss for each sequence."""
    embeddings_dict = {}
    loss_dict = {}
    # Create a DataLoader with consistent order (shuffle=False)
    data_loader = DataLoader(
        NormalizedSequenceDataset(normalized_sequences),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    autoencoder.model.eval()
    with torch.no_grad():
        # Track the original order of keys
        original_order = list(normalized_sequences.keys())
        original_index = 0
        
        for batch in tqdm(data_loader, desc="Applying Autoencoder and Calculating Loss", dynamic_ncols=True):
            batch_tensor = batch.to(device) 

            # Get reconstructed sequences and embeddings
            reconstructed_sequences = autoencoder.model(batch_tensor)
            encoded_representations = autoencoder.model.encoder(batch_tensor).cpu().numpy()

            # Calculate MSE loss for each sequence individually
            for i, reconstructed_seq in enumerate(reconstructed_sequences):
                original_len = original_lengths[original_index]

                # Calculate MSE loss between reconstructed and original sequence
                loss = mse_loss(
                    reconstructed_seq[:original_len], 
                   batch_tensor[i, :original_len, :]
                )             
                # Store the results with the corresponding key
                key = original_order[original_index]
                embeddings_dict[key] = encoded_representations[i]
                loss_dict[key] = loss.item()
                original_index += 1
    return embeddings_dict, loss_dict



#### ENCODING


basepath = '/LSTM'
# data_directory = os.path.join(basepath,'ExtractedTrips/l3harris/tripdata')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Loading data finished')
# Example dictionaries to process

dicts_to_process = {
    # 'speed2': speed2,
    # # # 'speed3': speed3,
    # 'detour03': detour03,
    # # 'detour05': detour05
    # 'backforth03': backforth03,
    # 'stop03': stop03,
    # 'stitch': stitch,
    # 'rotation': rotation,
    # 'repeat05': repeat05,
    # 'repeat1': repeat1,
    # 'loop': loop
    'train': train,
    'val': val,
    'anomalous': anomalous
}

# List of configurations for each model
model_configurations = [
    # {
    #     'name': 'route',
    #     'save_path': os.path.join(basepath, 'Model/{}/route_512'.format(dataset)),
    #     'input_size': 2,
    #     'hidden_size': 512,
    #     'checkpoint': 'model_epoch_18.pth'
    # },
    # {
    #     'name': 'route',
    #     'save_path': os.path.join(basepath, 'Model/{}/route_128'.format(dataset)),
    #     'input_size': 2,
    #     'hidden_size': 128,
    #     'checkpoint': 'model_epoch_100.pth'
    # },
    {
        'name': 'route',
        'save_path': os.path.join(basepath, 'Model/{}/route_64'.format(dataset)),
        'input_size': 2,
        'hidden_size': 64,
        'checkpoint': 'model_epoch_100.pth'
    },
    {
        'name': 'route',
        'save_path': os.path.join(basepath, 'Model/{}/route_16'.format(dataset)),
        'input_size': 2,
        'hidden_size': 16,
        'checkpoint': 'model_epoch_100.pth'
    },
    # Add more configurations as needed
]

# General training configuration common to all models
train_config = {
    'seed': 42,
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'device': device
}

# Function to set seeds
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load each model based on its configuration
for config in model_configurations:
    set_seed(train_config['seed'])
    modeltype = config['name']
    embsize = config['hidden_size']
    
    # Model specific configurations
    model_config = {
        'input_size': config['input_size'],
        'hidden_size': config['hidden_size'],
        'num_layers': 2,
    }
    
    # Update training config with model specific save_path
    train_config['save_path'] = config['save_path']
    
    # Initialize and load the model
    autoencoder = LSTMAutoencoder(model_config=model_config).to(train_config['device'])
    trainer = LSTMAutoencoderTrainer(model=autoencoder, train_loader=None, 
                                     val_loader=None, test_loader=None,
                                     train_config=train_config)
    trainer.load_checkpoint(os.path.join(config['save_path'], config['checkpoint']))
    print(f"Loaded model '{config['name']}' from {config['save_path']} with checkpoint {config['checkpoint']}")


    for dict_name, trip_dict in dicts_to_process.items():

        normalized_sequences = normalizedict(trip_dict)
        original_lengths = [len(seq) for seq in normalized_sequences.values()]
        embeddings, loss_dict = apply_autoencoder_and_calculate_loss(trainer,normalized_sequences,device,batch_size=128)


        with open(filepathemb, 'wb') as file:
            pickle.dump(embeddings, file)
        with open(filepathloss, 'wb') as file:
            pickle.dump(loss_dict, file)
        print(f"Finished encoding '{dict_name}' for '{config['name']}' from {config['save_path']}")

# Imports
import torch
import torch.optim as optim
from Data_utils import set_seed, create_data_loaders, encode_data_dict,filter_trips
from LSTM_model import LSTMAutoencoder, LSTMAutoencoderTrainer
import os
import pandas as pd
import numpy as np
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle

# FUNCTIONS

# SHAPE
dataset = 'Chengdu'
if dataset == 'Porto':
    datasetsaving = 'porto'
elif dataset == 'LA':
    datasetsaving = 'LA'
elif dataset == 'Chengdu':
    datasetsaving = 'Chengdu'
def normalizedict(datadict):
    # Relevant columns to extract
    relevant_columns = ['delta_x', 'delta_y']
    # Dictionary to hold normalized sequences
    normalized_sequences = {}
    # Extract relevant columns and normalize sequences
    for key, dataframe in datadict.items():
        # Extract relevant columns to create a numpy array
        sequence = np.array([dataframe[col].values for col in relevant_columns]).T
        # Store the normalized sequence in the dictionary
        normalized_sequences[key] = sequence
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
def apply_autoencoder_and_calculate_loss(autoencoder, normalized_sequences, device, batch_size=32):
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

basepath = '/ocean/projects/cis220071p/caos/LSTM'
# data_directory = os.path.join(basepath,'ExtractedTrips/l3harris/tripdata')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of files to encode
# data_directory = '/data/shurui/{}Synthetic/Processed'.format(dataset)
data_directory = '/ocean/projects/cis220071p/caos/LSTM/Chengdu/processed'
train = pd.read_pickle(os.path.join(data_directory, 'shape_train.pkl'))
train = filter_trips(train, 3, 150000)
val = pd.read_pickle(os.path.join(data_directory, 'shape_val.pkl'))
val = filter_trips(val, 3, 150000)
anomalous = pd.read_pickle(os.path.join(data_directory, 'shape_anomalous.pkl'))
anomalous = filter_trips(anomalous, 3, 150000)

# # List of files to encode
# data_directory = '/data/shurui/{}Synthetic/Processed'.format(dataset)
# # backforth03 = pd.read_pickle(os.path.join(data_directory, 'shape_backforth03.pickle'))
# # stop03 = pd.read_pickle(os.path.join(data_directory, 'shape_stop03.pickle'))
# # detour03 = pd.read_pickle(os.path.join(data_directory, 'shape_detour03.pickle'))
# # speed2 = pd.read_pickle(os.path.join(data_directory, 'shape_speed2.pickle'))
# # test = pd.read_pickle('/data/shurui/{}/processed/shape_test.pkl'.format(dataset))
# stitch = pd.read_pickle(os.path.join(data_directory, 'shape_stitch.pickle'))
# rotation = pd.read_pickle(os.path.join(data_directory, 'shape_rotation.pickle'))
# repeat05 = pd.read_pickle(os.path.join(data_directory, 'shape_repeat05.pickle'))
# repeat1 = pd.read_pickle(os.path.join(data_directory, 'shape_repeat1.pickle'))
# loop = pd.read_pickle(os.path.join(data_directory, 'shape_loop.pickle'))
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
    #     'name': 'shape',
    #     'save_path': os.path.join(basepath, 'Model/{}/shape_512'.format(dataset)),
    #     'input_size': 2,
    #     'hidden_size': 512,
    #     'checkpoint': 'model_epoch_92.pth'
    # },
    # {
    #     'name': 'shape',
    #     'save_path': os.path.join(basepath, 'Model/{}/shape_128'.format(dataset)),
    #     'input_size': 2,
    #     'hidden_size': 128,
    #     'checkpoint': 'model_epoch_100.pth'
    # },
    {
        'name': 'shape',
        'save_path': os.path.join(basepath, 'Model/{}/shape_64'.format(dataset)),
        'input_size': 2,
        'hidden_size': 64,
        'checkpoint': 'model_epoch_100.pth'
    },
    {
        'name': 'shape',
        'save_path': os.path.join(basepath, 'Model/{}/shape_16'.format(dataset)),
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

        filepathemb = '/ocean/projects/cis220071p/caos/LSTM/Extracted/{}/{}_embeddings_{}_{}_{}.pkl'.format(datasetsaving,modeltype,datasetsaving,dict_name,embsize)
        filepathloss = '/ocean/projects/cis220071p/caos/LSTM/Extracted/{}/{}_loss_{}_{}_{}.pkl'.format(datasetsaving,modeltype,datasetsaving,dict_name,embsize)
        with open(filepathemb, 'wb') as file:
            pickle.dump(embeddings, file)
        with open(filepathloss, 'wb') as file:
            pickle.dump(loss_dict, file)
        print(f"Finished encoding '{dict_name}' for '{config['name']}' from {config['save_path']}")

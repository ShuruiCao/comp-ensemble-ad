# Imports
import torch
import torch.optim as optim
from Data_utils import set_seed, create_data_loaders, encode_data_dict
from LSTM_model import LSTMAutoencoder, LSTMAutoencoderTrainer
import os
import pandas as pd
# torch.cuda.empty_cache()
basepath = '/LSTM'
# data_directory = os.path.join(basepath,'ExtractedTrips/l3harris/tripdata')


train = pd.read_pickle(os.path.join(data_directory, 'routespeed_train.pkl'))
val = pd.read_pickle(os.path.join(data_directory, 'routespeed_val.pkl'))
test = pd.read_pickle(os.path.join(data_directory, 'routespeed_anomalous.pkl'))
print('Loading data completed')
save_path = os.path.join(basepath,'Model/Chengdu/speed_128')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_config = {
    'batch_size': 64, # 64 when hiddensize is 64
    'dataset': 'Chengdu', # LA or porto
    'dataset_type': 'speed', # 'route' or 'speed' or 'shape'
    'normalize': True
}

model_config = {
    'input_size': 1,  # 2 for route or shape; 1 for speed
    'hidden_size': 128,
    'num_layers' : 2,
}

train_config = {
    'seed': 42,
    'save_path': save_path,
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'device':device
}

set_seed(train_config['seed'])
train_loader = create_data_loaders(data_config,train)
val_loader = create_data_loaders(data_config,val)
test_loader = create_data_loaders(data_config,test)
print('the number of training batches: ', len(train_loader))
print('take a look at train loader')
counter = 0
for batch in train_loader:
    for i, element in enumerate(batch):
        print(f"Element {i}: Type - {type(element)}, Size - {element.size()}")
    counter += 1
    if counter >= 2:
        break
print('Preprocessing complete...')
print(f'training with device: {device}')
autoencoder = LSTMAutoencoder(model_config=model_config).to(device)
trainer = LSTMAutoencoderTrainer(model=autoencoder, train_loader=train_loader, 
                                 val_loader=val_loader, test_loader=test_loader,
                                 train_config=train_config)
trainer.train(train_config['num_epochs'])

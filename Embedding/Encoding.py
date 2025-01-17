import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Dataset
import argparse
from Utils.Data_utils import set_seed
from Model.LSTM_model import LSTMAutoencoder, LSTMAutoencoderTrainer


class TripEncoder:
    def __init__(self, dataset, data_type, basepath, device, normalization_params, model_configs):
        self.dataset = dataset
        self.data_type = data_type
        self.basepath = basepath
        self.device = device
        self.normalization_params = normalization_params
        self.model_configs = model_configs
        self.data_directory = f'/data/shurui/{dataset}Synthetic/Processed'
        self.dicts_to_process = self.load_data()

    def load_data(self):
        """Load data dictionaries to process based on data_type."""
        filepath = f'{self.basepath}/Data/{self.dataset}/{self.dataset}_{self.data_type}_test.pkl'
        data = pd.read_pickle(filepath)
        return data

    def normalize_sequence(self, sequence):
        if self.data_type == 'speed':
            mean, std = self.normalization_params['mean'], self.normalization_params['std']
        elif self.data_type == 'shape':
            mean, std = 0, 1  # Shape doesn't require normalization
        elif self.data_type == 'route':
            mean = np.array([self.normalization_params['mean_lat'], self.normalization_params['mean_long']])
            std = np.array([self.normalization_params['std_lat'], self.normalization_params['std_long']])
        return (sequence - mean) / std

    def normalizedict(self, datadict):
        relevant_columns = {
            'speed': ['speed'],
            'shape': ['delta_x', 'delta_y'],
            'route': ['latitude', 'longitude']
        }[self.data_type]
        normalized_sequences = {}
        for key, dataframe in datadict.items():
            sequence = np.array([dataframe[col].values for col in relevant_columns]).T
            normalized_sequences[key] = self.normalize_sequence(sequence)
        return normalized_sequences

    def custom_collate_fn(self, batch):
        max_len = max(seq.shape[0] for seq in batch)
        padded_sequences = [np.pad(seq, ((0, max_len - seq.shape[0]), (0, 0)), mode='constant') for seq in batch]
        return torch.tensor(padded_sequences, dtype=torch.float32)

    def apply_autoencoder_and_calculate_loss(self, autoencoder, normalized_sequences, original_lengths, batch_size=32):
        embeddings_dict = {}
        loss_dict = {}
        data_loader = DataLoader(
            NormalizedSequenceDataset(normalized_sequences),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.custom_collate_fn
        )
        autoencoder.model.eval()
        with torch.no_grad():
            original_order = list(normalized_sequences.keys())
            original_index = 0
            for batch in tqdm(data_loader, desc="Encoding and Calculating Loss", dynamic_ncols=True):
                batch_tensor = batch.to(self.device)
                reconstructed_sequences = autoencoder.model(batch_tensor)
                encoded_representations = autoencoder.model.encoder(batch_tensor).cpu().numpy()
                for i, reconstructed_seq in enumerate(reconstructed_sequences):
                    original_len = original_lengths[original_index]  # Retrieve the original length
                    loss = mse_loss(
                        reconstructed_seq[:original_len],
                        batch_tensor[i, :original_len, :]
                    )
                    key = original_order[original_index]
                    embeddings_dict[key] = encoded_representations[i]
                    loss_dict[key] = loss.item()
                    original_index += 1
        return embeddings_dict, loss_dict

    def process(self):
        for config in self.model_configs:
            set_seed(42)
            autoencoder = LSTMAutoencoder(config).to(self.device)
            trainer = LSTMAutoencoderTrainer(model=autoencoder, train_loader=None, val_loader=None, test_loader=None, train_config=config)
            trainer.load_checkpoint(os.path.join(config['save_path'], config['checkpoint']))
            print(f"Loaded model {config['name']} from {config['save_path']}")

            for dict_name, trip_dict in self.dicts_to_process.items():
                normalized_sequences = self.normalizedict(trip_dict)
                original_lengths = [len(seq) for seq in normalized_sequences.values()]
                embeddings, loss_dict = self.apply_autoencoder_and_calculate_loss(trainer, normalized_sequences, original_lengths)
                emb_file = f'{self.basepath}/Extracted/{self.dataset}ver2/{config["name"]}_embeddings_{dict_name}.pkl'
                loss_file = f'{self.basepath}/Extracted/{self.dataset}ver2/{config["name"]}_loss_{dict_name}.pkl'
                with open(emb_file, 'wb') as ef, open(loss_file, 'wb') as lf:
                    pickle.dump(embeddings, ef)
                    pickle.dump(loss_dict, lf)
                print(f"Processed {dict_name} for model {config['name']}.")


class NormalizedSequenceDataset(Dataset):
    def __init__(self, data):
        self.data = list(data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Encode trips using LSTM autoencoder")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name (e.g., 'chengdu', 'LA', 'Porto')")
    parser.add_argument('--data_type', type=str, required=True, choices=['speed', 'shape', 'route'], help="Data type to process")
    parser.add_argument('--basepath', type=str, default='/home/shuruic/LSTM', help="Base path for data and models")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use (e.g., 'cuda:0', 'cpu')")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    normalization_params = {
        'speed': {'mean': 132.90087, 'std': 364.99795},
        'shape': None,
        'route': {'mean_lat': 0.00335, 'mean_long': 0.00288, 'std_lat': 0.13489, 'std_long': 0.11274}
    }[args.data_type]

    model_configs = [
        {'name': args.data_type, 'save_path': f'{args.basepath}/Model/{args.dataset}/{args.data_type}_128', 'input_size': 2, 'hidden_size': 128, 'checkpoint': 'model_epoch_100.pth'},
        {'name': args.data_type, 'save_path': f'{args.basepath}/Model/{args.dataset}/{args.data_type}_64', 'input_size': 2, 'hidden_size': 64, 'checkpoint': 'model_epoch_100.pth'}
    ]

    encoder = TripEncoder(
        dataset=args.dataset,
        data_type=args.data_type,
        basepath=args.basepath,
        device=torch.device(args.device),
        normalization_params=normalization_params,
        model_configs=model_configs
    )
    encoder.process()

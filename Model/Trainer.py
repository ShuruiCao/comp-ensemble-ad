import argparse
import os
import torch
import torch.optim as optim
from Utils.Data_utils import set_seed, create_data_loaders
from LSTM_model import LSTMAutoencoder, LSTMAutoencoderTrainer
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder for trajectory anomaly detection")
    parser.add_argument('--basepath', type=str, default='', 
                        help="Base path for data and model saving")
    parser.add_argument('--data_directory', type=str, default='/Data/porto/processed', 
                        help="Directory containing training, validation, and test data")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for data loaders")
    parser.add_argument('--dataset', type=str, default='porto', help="Dataset name")
    parser.add_argument('--data_type', type=str, default='shape', help="Type of dataset: One of 'route', 'speed', 'shape'")
    parser.add_argument('--normalize', action='store_true', help="Normalize data (default: False)")
    parser.add_argument('--input_size', type=int, default=2, help="Input size (2 for route/shape, 1 for speed)")
    parser.add_argument('--hidden_size', type=int, default=128, help="Hidden size of LSTM layers")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of LSTM layers")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--cuda_device', type=str, default='cuda:3', 
                        help="CUDA device to use (default: cuda:3)")
    return parser.parse_args()

def set_derived_parameters(args):
    # Automatically set `save_path` based on `hidden_size`
    args.save_path = f"Model/{args.dataset}/{args.data_type}_{args.hidden_size}"
    
    # Automatically set `num_layers` based on `data_type`
    dataa_type_to_num_layers = {'route': 2, 'speed': 1, 'shape': 2}
    args.num_layers = dataa_type_to_num_layers.get(args.data_type, 2)  # Default to 2 if not matched
    
    # Ensure save_path directory exists
    os.makedirs(args.save_path, exist_ok=True)
    
    return args


def main():
    args = parse_arguments()
    args = set_derived_parameters(args)
    
    # Setting up paths
    data_directory = args.data_directory
    save_path = os.path.join(args.basepath, args.save_path)
    os.makedirs(save_path, exist_ok=True)

    # Load data
    train = pd.read_pickle(os.path.join(data_directory, f'{args.data_type}_train.pkl'))
    val = pd.read_pickle(os.path.join(data_directory, f'{args.data_type}_val.pkl'))
    test = pd.read_pickle(os.path.join(data_directory, f'{args.data_type}_test.pkl'))
    print('Loading data completed')
    
    # Configure device
    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    print(f'Training with device: {device}')
    
    # Data configuration
    data_config = {
        'batch_size': args.batch_size,
        'dataset': args.dataset,
        'data_type': args.data_type,
        'normalize': args.normalize,
    }

    # Model configuration
    model_config = {
        'input_size': args.input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
    }

    # Training configuration
    train_config = {
        'seed': args.seed,
        'save_path': save_path,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'device': device,
    }

    # Set random seed
    set_seed(train_config['seed'])

    # Create data loaders
    train_loader = create_data_loaders(data_config, train)
    val_loader = create_data_loaders(data_config, val)
    test_loader = create_data_loaders(data_config, test)

    print('Data loaders created...')
    print(f'Training batches: {len(train_loader)}')

    # Initialize model and trainer
    autoencoder = LSTMAutoencoder(model_config=model_config).to(device)
    trainer = LSTMAutoencoderTrainer(model=autoencoder, train_loader=train_loader, 
                                     val_loader=val_loader, test_loader=test_loader, 
                                     train_config=train_config)
    
    # Uncomment to load a pre-trained checkpoint
    # trainer.load_checkpoint(f"{train_config['save_path']}/model_epoch_100.pth")

    # Train the model
    trainer.train(train_config['num_epochs'])

if __name__ == "__main__":
    main()

from torch import nn
import torch
import os
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers, 
            batch_first=True,
            dropout=0.0
        )
        
    def forward(self, x):
        # X shape [batch_size, seq_len, n_features]
        x, (hidden_n, _) = self.rnn(x)
        
        return hidden_n[-1]
    
class Decoder(nn.Module):
    def __init__(self, input_dim, n_features, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x, seq_len):
        x = x.unsqueeze(1).repeat(1, seq_len, 1)  # Repeat x for seq_len times
        x, _ = self.rnn(x)
        x = self.output_layer(x)  # Shape: [batch_size, seq_len, n_features]
        return x


class LSTMAutoencoder(nn.Module):
    def __init__(self, model_config):
        super(LSTMAutoencoder, self).__init__()
        n_features = model_config['input_size']
        embedding_dim = model_config['hidden_size']
        num_layers = model_config['num_layers']
        self.encoder = Encoder(n_features, embedding_dim,num_layers)
        self.decoder = Decoder(embedding_dim, n_features, embedding_dim, num_layers)
    def forward(self, x, return_embeddings=False):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, x.size(1))
        if return_embeddings:
            return decoded, encoded  # Return both reconstruction and embeddings
        else:
            return decoded


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

import torch.optim as optim
import os

class LSTMAutoencoderTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, train_config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_config = train_config
        self.best_val_loss = float('inf')
        self.save_path = train_config['save_path']
        self.loss_history = {'train': [], 'val': []}
        self.device = train_config['device']
        self.model.to(self.device)
        self.criterion = nn.MSELoss()  # Standard MSE Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_config['learning_rate'])

    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, leave=True)
        for batch in loop:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            recon = self.model(batch)
            loss = self.criterion(recon, batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        loop = tqdm(self.val_loader, leave=True)
        with torch.no_grad():
            for batch in loop:
                batch = batch.to(self.device)
                recon = self.model(batch)
                loss = self.criterion(recon, batch)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def test(self):
        self.model.eval()
        total_loss = 0
        loop = tqdm(self.test_loader, leave=True)
        with torch.no_grad():
            for batch in loop:
                batch = batch.to(self.device)

                recon = self.model(batch)
                loss = self.criterion(recon, batch)

                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        return avg_loss
    
    def train(self, num_epochs):

        for epoch in range(num_epochs):

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.loss_history['train'].append(train_loss)
            self.loss_history['val'].append(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch)

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        test_loss = self.test()
        print(f'test loss is {test_loss}')
        self.save_checkpoint(epoch)


    def save_checkpoint(self, epoch):
        filepath = f"{self.save_path}/model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'loss_history': self.loss_history
        }, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.loss_history = checkpoint['loss_history']
        self.best_val_loss = checkpoint['best_val_loss']



import torch
from dataset import load_dataset
from model import SolubilityModel
from train import Trainer

def main():
    # Hyperparameters
    in_channels = 4
    hidden_channels = 4
    out_channels = 1  # Regression task
    num_layers = 5
    dropout = 0.3
    heads = 8
    lr = 0.001
    weight_decay = 1e-5
    epochs = 50
    patience = 5
    batch_size = 32
    edge_dim = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, val_loader, test_loader = load_dataset(batch_size, 'delaney-processed.csv')

    # Initialize model
    model = SolubilityModel(in_channels, hidden_channels, out_channels, num_layers, dropout, heads, edge_dim)
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, lr, weight_decay, device)
    
    # Train model
    trainer.train(epochs, patience)

    # Test model
    trainer.test(test_loader)

if __name__ == "__main__":
    main()

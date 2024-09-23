import torch
import torch.nn as nn
import torch.optim as optim
from tools import EarlyStopping, rmse

class Trainer:
    def __init__(self, model, train_loader, val_loader, lr, weight_decay, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device

    def train(self, epochs, patience):
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for data in self.train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                output = output.squeeze(1)
                loss = self.criterion(output, data.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            val_loss = self.evaluate()
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    def evaluate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, data.y)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)

    def test(self, test_loader):
        self.model.load_state_dict(torch.load('checkpoint.pth'))
        self.model.eval()
        test_rmse = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                test_rmse += rmse(output, data.y)
        
        print(f'Test RMSE: {test_rmse / len(test_loader):.4f}')
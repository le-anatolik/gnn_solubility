import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from tqdm import tqdm
from dataset import MoleculeDataset
from model import GNN


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(train_loader):
        batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        all_preds.append(pred.cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    log_metrics(all_preds, all_labels, epoch, "train")
    
    return epoch_loss

def test(epoch, model, test_loader, loss_fn):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            pred = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
            loss = loss_fn(torch.squeeze(pred), batch.y.float())
            running_loss += loss.item()
            all_preds.append(pred.cpu().detach().numpy())
            all_labels.append(batch.y.cpu().detach().numpy())
    
    epoch_loss = running_loss / len(test_loader)
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    log_metrics(all_preds, all_labels, epoch, "test")
    
    return epoch_loss

def log_metrics(y_pred, y_true, epoch, stage):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nEpoch {epoch} | {stage.capitalize()} Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")
    
def run_one_training(params):
    params = params[0]
    with mlflow.start_run() as run:
        # Log parameters
        for key in params.keys():
            mlflow.log_param(key, params[key])

        # Load dataset
        print("Loading dataset...")
        train_dataset = MoleculeDataset(root="data/", filename='delaney-processed.csv')
        test_dataset = MoleculeDataset(root="data/", filename='delaney-processed.csv', test=True)
        params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]

        # Prepare data loaders
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

        # Initialize model
        print("Loading model...")
        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params)
        model = model.to(device)
        print(f"Number of parameters: {count_parameters(model)}")
        mlflow.log_param("num_params", count_parameters(model))

        # Set up loss function, optimizer, and scheduler
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"],
                                    momentum=params["sgd_momentum"],
                                    weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])

        # Training loop with early stopping
        best_loss = float('inf')
        early_stopping_counter = 0
        for epoch in range(300):
            if early_stopping_counter <= 10:
                # Training
                train_loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
                print(f"Epoch {epoch} | Train Loss {train_loss:.4f}")
                mlflow.log_metric(key="Train loss", value=train_loss, step=epoch)

                # Testing
                if epoch % 5 == 0:
                    test_loss = test(epoch, model, test_loader, loss_fn)
                    print(f"Epoch {epoch} | Test Loss {test_loss:.4f}")
                    mlflow.log_metric(key="Test loss", value=test_loss, step=epoch)
                    
                    # Early stopping
                    if test_loss < best_loss:
                        best_loss = test_loss
                        mlflow.pytorch.log_model(model, "model", signature=None)
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()
            else:
                print("Early stopping due to no improvement.")
                break
        print(f"Finished training with best test loss: {best_loss:.4f}")
        return [best_loss]

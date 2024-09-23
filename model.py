import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool

class SolubilityModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads, edge_dim):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Add subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))

        self.fc1 = nn.Linear(hidden_channels * heads, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index.long(), data.edge_attr, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(self.batch_norms[i](x))  # Batch normalization after each conv
            x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

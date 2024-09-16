import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_geometric.nn import TransformerConv, TopKPooling, global_mean_pool as gmp, global_max_pool as gap

class GNN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super().__init__()
        
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        self.dropout_rate = model_params["model_dropout_rate"]  
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]

        
        self.conv1 = TransformerConv(in_channels=feature_size, 
                                     out_channels=embedding_size, 
                                     heads=n_heads, 
                                     dropout=self.dropout_rate,
                                     edge_dim=edge_dim,
                                     beta=True)

        self.transf1 = Linear(in_features=embedding_size*n_heads, out_features=embedding_size)
        self.bn1 = BatchNorm1d(num_features=embedding_size)

        
        self.conv_layers = ModuleList([
            TransformerConv(in_channels=embedding_size, 
                            out_channels=embedding_size, 
                            heads=n_heads, 
                            dropout=self.dropout_rate,
                            edge_dim=edge_dim,
                            beta=True) for _ in range(self.n_layers)
        ])

        self.transf_layers = ModuleList([
            Linear(in_features=embedding_size*n_heads, out_features=embedding_size) for _ in range(self.n_layers)
        ])

        self.bn_layers = ModuleList([
            BatchNorm1d(num_features=embedding_size) for _ in range(self.n_layers)
        ])

        self.pooling_layers = ModuleList([
            TopKPooling(in_channels=embedding_size, ratio=top_k_ratio) for i in range(self.n_layers) if i % self.top_k_every_n == 0
        ])

        
        self.linear1 = Linear(in_features=embedding_size*2, out_features=dense_neurons)
        self.linear2 = Linear(in_features=dense_neurons, out_features=int(dense_neurons/2))
        self.linear3 = Linear(in_features=int(dense_neurons/2), out_features=1)

    def forward(self, x, edge_attr, edge_index, batch_index):
        
        x = self._apply_conv_block(x, edge_attr, edge_index, self.conv1, self.transf1, self.bn1)
        
        global_representation = []
        for i in range(self.n_layers):
            x = self._apply_conv_block(x, edge_attr, edge_index, 
                                       self.conv_layers[i], 
                                       self.transf_layers[i], 
                                       self.bn_layers[i])
            
            if i % self.top_k_every_n == 0 or i == self.n_layers - 1:
                x, edge_index, edge_attr, batch_index = self._apply_pooling(x, edge_index, edge_attr, batch_index, i)
                
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
        
        
        x = sum(global_representation)
        return self._apply_output_block(x)

    def _apply_conv_block(self, x, edge_attr, edge_index, conv_layer, transf_layer, bn_layer):
        x = conv_layer(x, edge_index, edge_attr)
        x = torch.relu(transf_layer(x))
        x = bn_layer(x)
        return x

    def _apply_pooling(self, x, edge_index, edge_attr, batch_index, i):
        pool_layer_idx = i // self.top_k_every_n
        x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[pool_layer_idx](x, edge_index, edge_attr, batch_index)
        return x, edge_index, edge_attr, batch_index

    def _apply_output_block(self, x):
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linear3(x)
        return x

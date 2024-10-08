import torch_geometric
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import os
from tqdm import tqdm

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.test = test
        self.filename = filename
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        with open(self.raw_paths[0]) as csvfile:
            self.data = csvfile.readlines()

        if self.test:
            return ['playing with the data']
        else:
            return [f'data_{i}.pt' for i in range(len(self.data)-1)]

    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0]) as csvfile:
            self.data = csvfile.readlines()
        self.smiles = [i.split(',')[-1] for i in self.data[1:]]
        self.labels = [float(i.split(',')[-2]) for i in self.data[1:]]
        for index, mol in enumerate(tqdm(self.smiles)):
            mol_obj = Chem.MolFromSmiles(mol)
            # Get node features
            node_feats = self._get_node_features(mol_obj)
            # Get edge features
            edge_feats = self._get_edge_features(mol_obj)
            # Get adjacency info
            edge_index = self._get_adjacency_info(mol_obj)
            # Get labels info
            label = self._get_labels(self.labels[index])

            # Create data object
            data = Data(x=node_feats, 
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        smiles=mol
                        ) 
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))

    def _get_node_features(self, mol):
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []    
            node_feats.append(atom.GetAtomicNum())
            node_feats.append(atom.GetHybridization())
            node_feats.append(atom.GetIsAromatic())
            node_feats.append(atom.GetTotalNumHs())
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Feature 3: Aromatic
            edge_feats.append(bond.GetIsAromatic())
            # Append edge features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]] #twice

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t()
        return torch.tensor(edge_indices, dtype=torch.int)

    def _get_labels(self, label):
        return torch.tensor(label, dtype=torch.float32)

    def len(self):
        return len(self.data)-1

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data

def load_dataset(batch_size, filename, root='data/'):
    
    dataset = MoleculeDataset(root=root, filename=filename)
    print(len(dataset))
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
from rdkit import Chem
from .featurizer import MoleculeFeaturizer
import torch

from torch_geometric.data import Data

from easydict import EasyDict
import yaml


class MolPropData(Data):
    def __init__(self, smiles):
        super().__init__()

        # Identification
        self.smiles: str = smiles
        
        # Viscosity-related information
        self.temps: torch.FloatTensor = None
        self.y: torch.FloatTensor = None
    
    
    def __repr__(self):
        return f'MolPropData(SMILES={self.smiles}, num_atomse={self.num_nodes}, edge_index={self.edge_index.shape})'   

def smiles2Data(smiles, temperature):
    mol = Chem.MolFromSmiles(smiles)
    data = MolPropData(smiles=smiles)
    data.temps = torch.tensor([temperature], dtype=torch.float32).unsqueeze(0)
    featurizer = MoleculeFeaturizer(additional_features=None)
    feature_dict = featurizer(mol)
    data.edge_index = feature_dict['edge_index']
    data.x = torch.tensor(feature_dict['x'], dtype=torch.float32)
    data.num_nodes = data.x.size(0)
    data.batch = torch.LongTensor([0]*data.num_nodes)
    return data


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

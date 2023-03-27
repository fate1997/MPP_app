import torch
from torch import nn
from .modules import GATv2Layer, ReadoutPhase, MLP, BoltzmannLayer


class MolPropModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.num_layers = config.model.num_layers  
        self.num_heads = config.model.num_heads
        self.dropout = config.model.dropout 
        self.num_energies = config.model.num_energies
        self.return_repr = config.model.return_repr
        
        self.message_passing = nn.ModuleList()
        for i in range(self.num_layers):
            self.message_passing.append(
                GATv2Layer(
                    self.input_dim if i == 0 else self.hidden_dim,
                    self.hidden_dim // self.num_heads,
                    self.num_heads,
                    dropout=self.dropout
                ))
        
        self.readout = ReadoutPhase(self.hidden_dim)

        
        self.pred_boltzmann = BoltzmannLayer(2*self.hidden_dim, self.num_energies)
        self.pred_surmount_prob = nn.Sigmoid()
        self.pred_extra = MLP(self.num_energies, 64, 1, 1, 0.1, nn.ELU())
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.message_passing:
            x = layer(x, edge_index)
        
        mol_repr = self.readout(x, data.batch)
        energy_distribution = self.pred_boltzmann(mol_repr, data.temps)
        output = self.pred_extra(energy_distribution).squeeze(-1)

        if self.return_repr:
            return x, mol_repr, energy_distribution, output
        else:
            return output
    

    def get_mol_repr(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.message_passing:
            x = layer(x, edge_index)
        
        mol_repr = self.readout(x, data.batch)
        return mol_repr
    
    def get_energy_dist(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.message_passing:
            x = layer(x, edge_index)
        
        mol_repr = self.readout(x, data.batch)
        energy_distribution = self.pred_boltzmann(mol_repr, data.temps)
        return energy_distribution
    
    def load_info(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path))
        return self
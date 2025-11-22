import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

class GIN_ESM_DTA(nn.Module):
    def __init__(self, n_atom_features=70, n_protein_features=480, hidden_dim=256, num_layers=4, dropout=0.2):
        super(GIN_ESM_DTA, self).__init__()

        # --- Stream A: Drug Encoder (Residual GIN) ---
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Dynamic Layer Generation
        for i in range(num_layers):
            input_dim = n_atom_features if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gin_layers.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # --- Stream B: Protein Encoder ---
        self.protein_projector = nn.Sequential(
            nn.Linear(n_protein_features, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Stream C: Interaction & Prediction ---
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        protein_emb = data.protein_emb

        # 1. GIN with Residual Connections
        h = x
        for i, (conv, bn) in enumerate(zip(self.gin_layers, self.batch_norms)):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # Residual connection (skip connection) if dims match
            # We can only skip if it's not the very first layer (which changes dim from 70->Hidden)
            if i > 0: 
                h = h + h_new 
            else:
                h = h_new
            
        drug_vector = global_add_pool(h, batch)

        # 2. Protein Stream
        protein_vector = self.protein_projector(protein_emb)

        # 3. Fusion
        cat_vector = torch.cat((drug_vector, protein_vector), dim=1)
        
        return self.predictor(cat_vector)

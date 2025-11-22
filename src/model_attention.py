import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import softmax

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (hidden_dim // heads) ** -0.5
        
        # Q (Protein), K (Drug Atoms), V (Drug Atoms)
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, drug_atoms, protein_ctx, batch_index):
        """
        drug_atoms: [Total_Atoms, Hidden] (The detailed graph features)
        protein_ctx: [Batch_Size, Hidden] (The global protein context)
        batch_index: [Total_Atoms] (Mapping which atom belongs to which batch item)
        """
        # 1. Expand Protein Context to match Atoms
        # If Batch=0 has 20 atoms, we repeat Protein[0] 20 times.
        protein_expanded = protein_ctx[batch_index] 
        
        # 2. Linear Projections
        Q = self.to_q(protein_expanded) # Protein is looking
        K = self.to_k(drug_atoms)       # Atoms are being looked at
        V = self.to_v(drug_atoms)       # Atoms offering information
        
        # 3. Calculate Attention Scores
        # Simple Dot Product Attention
        # Shape: [Total_Atoms, 1] (simplified for single head logic)
        dots = (Q * K).sum(dim=-1, keepdim=True) * self.scale
        
        # 4. Softmax (normalized per graph)
        # This ensures attention sums to 1 *per molecule*
        attn_weights = softmax(dots, batch_index)
        
        # 5. Weighted Sum
        # This effectively performs the "Pooling" step, but weighted by relevance!
        out = attn_weights * V
        
        # Residual Connection + Norm
        out = out + drug_atoms
        return self.norm(out), attn_weights

class GIN_ESM_Attention(nn.Module):
    def __init__(self, n_atom_features=70, n_protein_features=480, hidden_dim=256, num_layers=4, dropout=0.2):
        super().__init__()

        # --- Drug Encoder (GIN) ---
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            input_dim = n_atom_features if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gin_layers.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # --- Protein Encoder ---
        self.protein_projector = nn.Sequential(
            nn.Linear(n_protein_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # --- THE UPGRADE: Cross Attention ---
        self.attention = CrossAttentionBlock(hidden_dim)

        # --- Predictor ---
        # Note: Input is Hidden*2 because we concat (Pooled_Drug + Protein)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        protein_emb = data.protein_emb

        # 1. GIN (Get Atom Embeddings)
        h = x
        for i, (conv, bn) in enumerate(zip(self.gin_layers, self.batch_norms)):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            if i > 0: h = h + h_new 
            else: h = h_new
        
        # h is now [Total_Atoms, Hidden] - aka "Unpooled Features"

        # 2. Protein Project
        protein_vec = self.protein_projector(protein_emb) # [Batch, Hidden]

        # 3. Cross Attention (Interaction)
        # We pass the unpooled atoms (h) and let the protein "attend" to them
        attended_atoms, weights = self.attention(h, protein_vec, batch)
        
        # 4. Attention-Aware Pooling
        # Instead of generic sum/mean, we sum the attended features
        drug_vec = global_add_pool(attended_atoms, batch)

        # 5. Fusion & Predict
        cat_vector = torch.cat((drug_vec, protein_vec), dim=1)
        return self.predictor(cat_vector)

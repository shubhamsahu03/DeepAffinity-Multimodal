import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from joblib import Parallel, delayed
from tqdm import tqdm
from .features import get_atom_features, get_bond_features

def smile_to_graph_safe(smile, label, seq_embedding):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None: return None
        
        x = []
        for atom in mol.GetAtoms():
            x.append(get_atom_features(atom))
        x = torch.stack(x)
        
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            edge_index.append([u, v])
            edge_index.append([v, u])
            attr = get_bond_features(bond)
            edge_attr.append(attr)
            edge_attr.append(attr) 
            
        if len(edge_index) == 0: return None 
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # LOG TRANSFORMATION (Crucial for RMSE)
        val = float(label)
        if val < 1e-9: val = 1e-9 
        pkd = -np.log10(val * 1e-9)
        
        data.y = torch.tensor([[pkd]], dtype=torch.float)
        data.protein_emb = seq_embedding.unsqueeze(0) 
        
        return data
    except Exception:
        return None

def process_parallel(df, embedding_dict, n_jobs=-1):
    tasks = []
    dummy_emb = torch.zeros(480) # ESM-35M has dim 480 (not 320!)
    
    for idx, row in df.iterrows():
        seq = row['Target'] 
        emb = embedding_dict.get(seq, dummy_emb)
        tasks.append((row['Drug'], row['Y'], emb))
        
    results = Parallel(n_jobs=n_jobs)(
        delayed(smile_to_graph_safe)(smile, y, emb) for smile, y, emb in tqdm(tasks, desc="Graph Conversion")
    )
    return [d for d in results if d is not None]

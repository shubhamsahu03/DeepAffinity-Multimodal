import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in loader: # Removed tqdm for cleaner Optuna logs
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward
        pred = model(batch)
        label = batch.y.view(-1, 1) 
        
        # Loss & Backprop
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0
    if criterion is None: criterion = nn.MSELoss()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            pred = model(batch)
            label = batch.y.view(-1, 1)
            
            loss = criterion(pred, label)
            total_loss += loss.item() * batch.num_graphs
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    mse = total_loss / len(loader.dataset)
    rmse = np.sqrt(mse)
    
    if len(all_preds) > 1:
        spearman = spearmanr(all_labels, all_preds)[0]
    else:
        spearman = 0
        
    return {
        "mse": mse,
        "rmse": rmse,
        "spearman": spearman
    }

import torch
import numpy as np
from tqdm import tqdm
from scipy import stats

def enable_dropout(model):
    """
    Sets Dropout layers to train mode (active) while keeping
    Batch Norm and other layers in eval mode (frozen).
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def predict_mc_dropout(model, loader, device, n_repeats=30, confidence_level=0.95):
    """
    Runs inference 'n_repeats' times with active dropout to estimate uncertainty.
    
    Returns:
        mean_preds: The 'Ensemble' prediction (Most accurate)
        std_preds:  The Standard Deviation (Uncertainty score)
        ci_lower:   Lower bound of 95% Confidence Interval
        ci_upper:   Upper bound of 95% Confidence Interval
    """
    model.eval()        # 1. Freeze Batch Norm and weights
    enable_dropout(model) # 2. Force Dropout layers to remain active
    
    # We store predictions on CPU to prevent GPU OOM on large datasets
    all_preds_runs = [] 
    
    print(f"Running Monte Carlo Inference ({n_repeats} stochastic passes)...")
    
    # Outer loop: Number of MC passes
    for i in tqdm(range(n_repeats), desc="MC Dropout Sampling"):
        preds_single_run = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                
                # Forward pass (Dropout is randomly killing neurons here)
                output = model(batch)
                
                # Move to CPU immediately
                preds_single_run.extend(output.cpu().numpy().flatten())
        
        all_preds_runs.append(preds_single_run)
    
    # Convert to numpy array: Shape [n_repeats, n_samples]
    all_preds_runs = np.array(all_preds_runs)
    
    # --- Statistical Analysis ---
    # 1. Mean Prediction (The Ensemble Result)
    mean_preds = np.mean(all_preds_runs, axis=0) 
    
    # 2. Uncertainty (Standard Deviation)
    std_preds = np.std(all_preds_runs, axis=0)   
    
    # 3. 95% Confidence Interval calculation
    sem = stats.sem(all_preds_runs, axis=0)
    t_score = stats.t.ppf((1 + confidence_level) / 2, n_repeats - 1)
    
    margin_of_error = t_score * sem
    ci_lower = mean_preds - margin_of_error
    ci_upper = mean_preds + margin_of_error
    
    return mean_preds, std_preds, ci_lower, ci_upper

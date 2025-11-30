import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr, spearmanr

from src.model_attention import GIN_ESM_Attention
from src.dataset import BindingDBDataset
from src.uncertainty import predict_mc_dropout
from torch_geometric.loader import DataLoader

# --- CONFIGURATION ---
MODEL_PATH = "final_attention_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64   # Updated to match your training batch size
N_MC_PASSES = 50 

# --- 1. SETUP ---
print(f"--- ACTIVE LEARNING & METRICS PIPELINE ---")
print(f">>> Loading Model: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model weights not found! Please run training first.")

# Load Test Data
test_dataset = BindingDBDataset(root="./data", split_name='test')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize Model 
sample = test_dataset[0]
n_atom = sample.x.shape[1]
n_prot = sample.protein_emb.shape[1]

# --- THE FIX ---
# Initialization using YOUR specific training parameters
model = GIN_ESM_Attention(
    n_atom_features=n_atom,
    n_protein_features=n_prot,
    hidden_dim=128,        # Matched to your params
    num_layers=4,          # Matched to your params
    dropout=0.32207929065429397        # Matched to your params
).to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(">>> Model Loaded Successfully.")
except RuntimeError as e:
    print("\nCRITICAL ERROR: Model Architecture Mismatch")
    print("The code parameters do not match the saved file parameters.")
    print(e)
    exit()

# --- 2. RUN ROBUST UNCERTAINTY ESTIMATION ---
print(">>> Starting Monte Carlo Inference...")
means, stds, ci_lower, ci_upper = predict_mc_dropout(model, test_loader, DEVICE, n_repeats=N_MC_PASSES)

# --- 3. DATA ANALYSIS ---
actual_vals = np.array([data.y.item() for data in test_dataset])

df_results = pd.DataFrame({
    'Actual_pKd': actual_vals,
    'Predicted_pKd': means,
    'Uncertainty_Std': stds,
    'CI_Lower': ci_lower,
    'CI_Upper': ci_upper
})

# Calculate Error Metrics
df_results['Abs_Error'] = abs(df_results['Predicted_pKd'] - df_results['Actual_pKd'])

# --- 4. CALCULATE METRICS ---
acc_pearson, _ = pearsonr(df_results['Predicted_pKd'], df_results['Actual_pKd'])
acc_spearman, _ = spearmanr(df_results['Predicted_pKd'], df_results['Actual_pKd'])
rmse = np.sqrt(((df_results['Predicted_pKd'] - df_results['Actual_pKd']) ** 2).mean())
safety_pearson, _ = pearsonr(df_results['Uncertainty_Std'], df_results['Abs_Error'])

# --- 5. REPORTING ---
print("\n" + "="*60)
print(f"📊 FINAL MODEL PERFORMANCE REPORT")
print("="*60)
print(f"1. ACCURACY")
print(f"   - RMSE:              {rmse:.4f}")
print(f"   - Spearman (Rank):   {acc_spearman:.4f}")
print(f"   - Pearson (Linear):  {acc_pearson:.4f}")

print(f"\n2. SAFETY (Calibration)")
print(f"   - Uncertainty vs Error Pearson: {safety_pearson:.4f}")
if safety_pearson > 0.3:
    print("     ✅ PASS: System correctly assigns High Uncertainty to High Error.")
else:
    print("     ⚠️ WARNING: Calibration is weak.")
print("="*60)

# --- 6. VISUALIZATION ---
plt.figure(figsize=(16, 6))

# Plot 1: Accuracy
plt.subplot(1, 2, 1)
plt.scatter(df_results['Actual_pKd'], df_results['Predicted_pKd'], 
            alpha=0.5, c=df_results['Uncertainty_Std'], cmap='viridis')
plt.plot([2, 10], [2, 10], 'r--', lw=2)
plt.colorbar(label='Uncertainty (Std)')
plt.title(f"Accuracy: Pred vs Actual\nPearson r={acc_pearson:.3f}")
plt.xlabel("Actual pKd")
plt.ylabel("Predicted pKd")

# Plot 2: Safety
plt.subplot(1, 2, 2)
sns.regplot(data=df_results, x='Uncertainty_Std', y='Abs_Error', 
            scatter_kws={'alpha':0.3, 'color':'purple'}, line_kws={'color':'red'})
plt.title(f"Safety: Error vs Uncertainty\nPearson r={safety_pearson:.3f}")
plt.xlabel("Model Uncertainty (Std Dev)")
plt.ylabel("Prediction Error")

plt.tight_layout()
plt.savefig('final_comprehensive_analysis.png', dpi=300)
print("\n✅ Analysis saved to 'final_comprehensive_analysis.png'")

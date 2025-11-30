# 🧬 Graph-Attention-DTA: Multimodal Drug Discovery

![Status](https://img.shields.io/badge/Status-Completed-success)
![Metric](https://img.shields.io/badge/Spearman-0.788-brightgreen)
![Framework](https://img.shields.io/badge/PyTorch-Geometric-red)
![License](https://img.shields.io/badge/License-MIT-blue)

A Deep Learning framework for **Drug-Target Affinity (DTA)** prediction. This project solves the "Cold-Start" problem in drug discovery by using a **Multimodal Late-Fusion Architecture** that combines molecular graphs (Drugs) and protein sequences (Targets).

By implementing a **Cross-Attention Mechanism** between the drug atoms and protein amino acids, the model achieves high interpretability and robust generalization on unseen molecular scaffolds.

---

## 📊 Key Results
Evaluated on the **BindingDB Cold-Drug Split** (hardest benchmark), where the test set contains drug scaffolds never seen during training:

| Metric | Score | Industry Context |
| :--- | :--- | :--- |
| **Spearman Rank Correlation** | **0.788** | High ranking ability (SOTA tier for Cold Split) |
| **RMSE** | **0.708** | Precise affinity prediction (< 1 log unit error) |
| **Pearson Correlation** | **0.7982** | Strong linear relationship |

### Performance Visualization
![Results](https://github.com/shubhamsahu03/DeepAffinity-Multimodal/blob/streamlitv2/results/final_results.png)
*(Scatter plot showing predicted vs. actual pKd values on the test set. The tight clustering along the diagonal demonstrates low variance and high precision.)*

---

## 🧠 Model Architecture

The system uses a **Dual-Stream Encoder** with a Cross-Attention fusion head:

### 1. Drug Stream (Graph Neural Network)
* **Input:** SMILES string converted to 2D Molecular Graph.
* **Featurization:** Rich atom features (Symbol, Hybridization, Chirality, Aromaticity) + Bond types.
* **Model:** **Deep Residual GIN (Graph Isomorphism Network)**.
    * *Why GIN?* It is theoretically proven to be maximally discriminative for distinguishing graph structures (isomers).
    * *Why Residual?* Skip connections allow for a deeper network (5 layers) without vanishing gradients.

### 2. Protein Stream (Transformer)
* **Input:** Amino Acid Sequence.
* **Model:** **ESM-2 (Evolutionary Scale Modeling)** by Meta AI.
* **Configuration:** 35M Parameter version (`esm2_t12_35M_UR50D`).
* **Method:** We use the pre-trained transformer as a feature extractor to capture evolutionary binding patterns.

### 3. Fusion Stream (Cross-Attention)
* **Mechanism:** Multi-Head Cross-Attention.
* **Logic:** The Protein embedding acts as the `Query`, attending to specific Drug Atoms (`Keys/Values`).
* **Benefit:** This mimics "Induced Fit" theory, where the protein focuses on specific pharmacophores (e.g., a nitrogen ring) rather than the whole molecule equally.

---

## 🛠️ Engineering & MLOps

This project goes beyond standard tutorials by incorporating rigorous engineering practices:

* **Bayesian Optimization:** Used **Optuna** (TPE Sampler) to tune 5 hyperparameters (LR, Dropout, Layers, Hidden Dim, Batch Size).
    * *Discovery:* Found a high-regularization regime (**Dropout = 0.32**) was critical for generalization.
* **Parallel Data Pipeline:** Implemented a multi-core `joblib` processor to convert SMILES to Graphs, reducing preprocessing time by 90%.
* **Robust Evaluation:** Used **Cold-Drug Splitting** instead of random splitting to prevent data leakage and simulate real-world screening scenarios.

---

## 🚀 Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch 2.0+
* CUDA (Recommended)

### Setup
```bash
# Clone the repository
git clone [https://github.com/yourusername/Graph-Attention-DTA.git](https://github.com/yourusername/Graph-Attention-DTA.git)
cd Graph-Attention-DTA
```
## Install dependencies
```bash
pip install torch torch-geometric rdkit-pypi PyTDC transformers optuna matplotlib scipy
```

## 1. Process Data & Train the Model
```bash
python build_pipeline.py   # Processes BindingDB data
python src/train.py        # Runs training loop with best params
```
## 2. Run Inference on New Drugs
```bash
python inference.py --drug "CC(=O)OC1=CC=CC=C1C(=O)O" --target "MLP..."
```
```bash
📂 Project Structure
├── app/
|   ├── app.py
├── data/                  # Cached graph datasets and ESM embeddings
├── src/
│   ├── features.py        # Atom/Bond featurization logic
│   ├── preprocess.py      # ESM-2 embedding generation
│   ├── processor.py       # Parallel Graph conversion & pKd scaling
│   ├── dataset.py         # PyTorch Geometric InMemoryDataset
│   ├── model_attention.py # GIN + ESM + CrossAttention Architecture
|   ├── uncertainity.py    
|   ├── run_active_learning.py # Active Learning
│   └── train.py           # Training loop & Evaluation metrics
├── final_attention_model.pth # Trained Model Weights
├── Dockerfile  
└── README.md
```
## 🔮 3. Future Improvements

- **3D Geometric Deep Learning:** Replace the 2D GIN with an E(n)-Equivariant GNN (like EGNN or SchNet) to leverage 3D conformer coordinates, capturing spatial steric clashes.
- **Structure-Based Protein Encoder:** Utilize AlphaFold generated PDB structures with a GearNet (Geometry Aware GNN) instead of 1D sequences for better binding pocket analysis.

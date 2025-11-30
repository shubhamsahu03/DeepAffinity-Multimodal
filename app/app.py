import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, EsmModel
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from torch_geometric.data import Data
import sys
import os
import io
import urllib.parse

# --- 1. CONFIGURATION & PATHS ---
# Add project root to path to allow importing 'src'
# This is crucial for Docker/Hugging Face environments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.features import get_atom_features, get_bond_features
    from src.model_attention import GIN_ESM_Attention
except ImportError as e:
    st.error(f"🚨 Critical Error: Could not import 'src' modules. Ensure the 'src/' folder is in the root directory. Details: {e}")
    st.stop()

# --- MODEL CONFIGURATION (Must match 'final_attention_model.pth') ---
MODEL_PARAMS = {
    "hidden_dim": 128,      
    "num_layers": 4,        
    "dropout": 0.32207929065429397,       
    "n_atom_features": 79,  
    "n_protein_features": 480 
}

DEVICE = torch.device("cpu") # Use CPU for deployment to avoid OOM on free tiers
MODEL_PATH = "final_attention_model.pth"
ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"

# --- 2. PAGE CONFIG ---
st.set_page_config(
    page_title="DeepAffinity | AI Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50;}
    .stButton>button {width: 100%; border-radius: 8px; height: 3em; font-weight: bold;}
    .metric-container {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    </style>
    """, unsafe_allow_html=True)

# --- 3. BACKEND LOGIC (Cached) ---
@st.cache_resource
def load_components():
    """Loads model components once and caches them."""
    status = st.sidebar.empty()
    status.info("⏳ Initializing AI Engine... (This happens once)")
    
    try:
        # 1. Load Tokenizer & Protein Encoder
        tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(DEVICE)
        esm_model.eval()

        # 2. Initialize Architecture
        model = GIN_ESM_Attention(
            n_atom_features=MODEL_PARAMS["n_atom_features"],
            n_protein_features=MODEL_PARAMS["n_protein_features"],
            hidden_dim=MODEL_PARAMS["hidden_dim"],
            num_layers=MODEL_PARAMS["num_layers"],
            dropout=MODEL_PARAMS["dropout"]
        ).to(DEVICE)
        
        # 3. Load Weights
        if not os.path.exists(MODEL_PATH):
            # Fallback check for Docker path
            if os.path.exists(f"/app/{MODEL_PATH}"):
                path_to_load = f"/app/{MODEL_PATH}"
            else:
                st.error(f"🚨 Model file '{MODEL_PATH}' not found! Ensure it is in the root directory.")
                st.stop()
        else:
            path_to_load = MODEL_PATH
            
        state_dict = torch.load(path_to_load, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        
        status.success("✅ Engine Ready")
        return tokenizer, esm_model, model
        
    except Exception as e:
        st.error(f"🚨 System Startup Failed: {e}")
        st.stop()

def preprocess(smiles, target_seq, tokenizer, esm_model):
    """Robust preprocessing with error checks."""
    if not smiles or not target_seq: return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    try:
        # Protein Embedding
        inputs = tokenizer(target_seq, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = esm_model(**inputs)
            protein_emb = outputs.last_hidden_state.mean(dim=1)
            
        # Graph Construction
        x = []
        for atom in mol.GetAtoms(): x.append(get_atom_features(atom))
        x = torch.stack(x)
        
        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[u, v], [v, u]])
            attr = get_bond_features(bond)
            edge_attr.extend([attr, attr])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.protein_emb = protein_emb
        data.batch = torch.zeros(x.shape[0], dtype=torch.long)
        return data
    except Exception as e:
        return None

def predict_hybrid(model, data, n_passes=20):
    """Hybrid Inference: Deterministic Prediction + Stochastic Uncertainty."""
    # Deterministic Pass (Best RMSE)
    model.eval()
    with torch.no_grad(): pred = model(data).item()
    
    # Stochastic Pass (Uncertainty)
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'): m.train()
    
    noisy = []
    with torch.no_grad():
        for _ in range(n_passes): noisy.append(model(data).item())
        
    return pred, np.std(noisy)

def pkd_to_nm(pkd): 
    """Converts pKd to nM concentration."""
    return (10**(-pkd)) * 1e9

def get_props(smiles):
    """Calculates chemical properties."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return {}
    return {
        "MW": round(Descriptors.MolWt(mol), 1),
        "LogP": round(Descriptors.MolLogP(mol), 1),
        "H-Donors": Descriptors.NumHDonors(mol),
        "H-Acceptors": Descriptors.NumHAcceptors(mol)
    }

def draw_mol(smiles):
    """Draws 2D molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol: return Draw.MolToImage(mol, size=(450, 250))
    return None

def plot_radar_chart(props):
    """Generates a Drug-Likeness Radar Chart."""
    labels = ['MolWt', 'LogP', 'H-Donors', 'H-Acceptors']
    
    # Safely handle missing keys
    mw = props.get('MW', 0)
    logp = props.get('LogP', 0)
    hd = props.get('H-Donors', 0)
    ha = props.get('H-Acceptors', 0)

    # Normalize (Approximate Max Values: MW=500, LogP=5, HD=5, HA=10)
    values = [
        min(mw / 500, 1.0), 
        min(max(logp, 0) / 5, 1.0), 
        min(hd / 5, 1.0), 
        min(ha / 10, 1.0)
    ]
    
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='teal', alpha=0.25)
    ax.plot(angles, values, color='teal', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.spines['polar'].set_visible(False)
    ax.grid(color='gray', alpha=0.2)
    return fig

# --- 4. APP UI ---

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
st.sidebar.title("DeepAffinity")
st.sidebar.markdown("**AI-Powered Drug Discovery**")
mc_passes = st.sidebar.slider("MC Uncertainty Samples", 10, 50, 20, 
    help="Higher samples = more accurate confidence estimation, but slower.")
st.sidebar.info("Model: GIN-Attention + ESM-2 (35M)")

# Load System
tokenizer, esm_model, model = load_components()

# Main Header
st.title("🧬 DeepAffinity")
st.markdown("### AI-Powered Virtual Screening & Uncertainty Quantification")

tab1, tab2, tab3 = st.tabs(["🧪 Single Interaction", "📂 Batch Screening", "🧠 Model Info"])

# === TAB 1: SINGLE PREDICTION ===
with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("1. Interaction Setup")
        
        # Session state for persistence
        if 'smi' not in st.session_state: st.session_state.smi = ""
        if 'seq' not in st.session_state: st.session_state.seq = ""

        if st.button("🧪 Load Example: Imatinib + ABL1"):
            st.session_state.smi = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
            st.session_state.seq = "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEY"
            st.experimental_rerun()

        smiles = st.text_area("Drug Molecule (SMILES)", value=st.session_state.smi, height=100)
        prot = st.text_area("Target Protein (Sequence)", value=st.session_state.seq, height=150)
        
        btn_predict = st.button("🚀 Run Analysis", type="primary")

    with col2:
        st.subheader("2. Insight Dashboard")
        
        if btn_predict and smiles and prot:
            with st.spinner("Running Geometric Deep Learning..."):
                data = preprocess(smiles, prot, tokenizer, esm_model)
                
                if data is None:
                    st.error("❌ Invalid Input: Could not process SMILES or Protein Sequence.")
                else:
                    pkd, sigma = predict_hybrid(model, data, mc_passes)
                    nm_conc = pkd_to_nm(pkd)
                    props = get_props(smiles)
                    
                    # --- VISUALIZATION ROW ---
                    v1, v2 = st.columns([2, 1])
                    with v1:
                        st.image(draw_mol(smiles), caption="Ligand 2D Structure", use_column_width=True)
                    with v2:
                        st.pyplot(plot_radar_chart(props))
                    
                    # --- METRICS ROW ---
                    st.markdown("##### 📊 Prediction Metrics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted pKd", f"{pkd:.2f}")
                    m2.metric("Dissociation", f"{nm_conc:.1f} nM")
                    
                    lbl, clr = ("High Confidence", "normal") if sigma < 0.2 else ("Uncertain", "inverse")
                    m3.metric("Uncertainty", f"± {sigma:.3f}", delta=lbl, delta_color=clr)
                    
                    # --- INTERPRETATION ---
                    st.markdown("---")
                    if pkd > 7.0:
                        st.success("✅ **Strong Binder:** Likely effective at low concentrations.")
                    elif pkd > 5.0:
                        st.warning("⚠️ **Moderate Binder:** May require lead optimization.")
                    else:
                        st.error("❌ **Weak Binder:** Unlikely to be effective.")
                        
                    # --- EXTERNAL LINKS ---
                    st.markdown("##### 🌐 External Database Search")
                    l1, l2 = st.columns(2)
                    safe_smi = urllib.parse.quote(smiles)
                    l1.link_button("Search PubChem", f"https://pubchem.ncbi.nlm.nih.gov/#query={safe_smi}")
                    l2.link_button("BLAST Protein", f"https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastp&QUERY={prot[:50]}...")

# === TAB 2: BATCH SCREENING ===
with tab2:
    st.subheader("High-Throughput Screening")
    b_prot = st.text_area("Target Sequence", value=st.session_state.seq, height=100, key="batch_prot")
    up_file = st.file_uploader("Upload CSV (Must have 'SMILES' column)", type=['csv'])
    
    if up_file and b_prot and st.button("Start Screening"):
        try:
            df = pd.read_csv(up_file)
        except:
            st.error("Invalid CSV file.")
            st.stop()

        if 'SMILES' not in df.columns:
            st.error("CSV must have 'SMILES' column")
        else:
            results = []
            bar = st.progress(0)
            status = st.empty()
            
            for i, row in df.iterrows():
                status.text(f"Processing {i+1}/{len(df)}...")
                d = preprocess(row['SMILES'], b_prot, tokenizer, esm_model)
                if d:
                    p, s = predict_hybrid(model, d, n_passes=min(mc_passes, 10))
                    results.append({
                        "SMILES": row['SMILES'],
                        "Predicted_pKd": round(p, 3),
                        "Kd_nM": round(pkd_to_nm(p), 2),
                        "Uncertainty": round(s, 3)
                    })
                else:
                    results.append({"SMILES": row['SMILES'], "Predicted_pKd": None})
                bar.progress((i+1)/len(df))
            
            status.success("✅ Screening Complete!")
            res_df = pd.DataFrame(results).sort_values(by="Predicted_pKd", ascending=False)
            
            st.divider()
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", csv, "screening_results.csv", "text/csv")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("##### Affinity Distribution")
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.histplot(res_df['Predicted_pKd'].dropna(), kde=True, color='teal', ax=ax)
                ax.set_xlabel("Predicted pKd")
                st.pyplot(fig)
            with c2:
                st.markdown("##### Top Hits")
                st.dataframe(res_df.head(10), use_container_width=True)

# === TAB 3: MODEL INFO ===
with tab3:
    st.header("🧠 Model Architecture: Graph-Attention-DTA")
    st.markdown("""
    This model solves the **Cold-Start Problem** in drug discovery using a multimodal architecture.
    
    ### 1. The Architecture
    * **Drug Encoder:** Deep Residual Graph Isomorphism Network (GIN) captures molecular geometry and isomerism.
    * **Protein Encoder:** ESM-2 (Evolutionary Scale Modeling) Transformer (35M params) captures sequence evolutionary patterns.
    * **Fusion:** Multi-Head Cross-Attention allows the protein to "focus" on specific atoms in the drug.
    
    ### 2. Uncertainty Quantification
    * **Method:** Monte Carlo (MC) Dropout.
    * **Logic:** We keep dropout layers active during inference and run 20+ stochastic passes.
    * **Interpretation:** * **Low Variance:** The model is confident (Safe).
        * **High Variance:** The model is guessing (Unsafe). This flags "Out-of-Distribution" compounds.
        
    ### 3. Performance (Cold-Drug Split)
    * **Spearman Rank:** 0.778 (SOTA Tier)
    * **RMSE:** 0.716 (High Precision)
    """)
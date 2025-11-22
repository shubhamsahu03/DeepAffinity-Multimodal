import torch
import os
import pickle
from transformers import EsmModel, AutoTokenizer
from tqdm import tqdm

def cache_protein_embeddings(df, output_path, model_name="facebook/esm2_t12_35M_UR50D"):
    """
    Generates embeddings using the 35M parameter ESM-2 model.
    Higher capacity = Better protein representation.
    """
    if os.path.exists(output_path):
        print(f"Found cached embeddings at {output_path}")
        with open(output_path, "rb") as f:
            return pickle.load(f)
            
    print(f"Generating embeddings using {model_name} (High Quality)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    unique_seqs = df['Target'].unique().tolist()
    embedding_dict = {}
    batch_size = 16 
    
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_seqs), batch_size), desc="Embedding Proteins"):
            batch_seqs = unique_seqs[i:i+batch_size]
            
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
            
            for seq, emb in zip(batch_seqs, embeddings):
                embedding_dict[seq] = emb
            
    with open(output_path, "wb") as f:
        pickle.dump(embedding_dict, f)
        
    return embedding_dict

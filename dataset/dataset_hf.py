import os
import json
from datasets import Dataset

DESCRIPTIONS_DIR = "dataset/descriptions"
VGDL_DIR = "dataset/vgdl_files"

def load_pairs():
    pairs = []
    desc_files = sorted(os.listdir(DESCRIPTIONS_DIR))
    
    for fname in desc_files:
        stem = os.path.splitext(fname)[0]
        desc_path = os.path.join(DESCRIPTIONS_DIR, fname)
        
        # cerca il file VGDL corrispondente (prova estensioni diverse)
        vgdl_path = None
        for ext in [".vgdl", ".txt", ".py"]:
            candidate = os.path.join(VGDL_DIR, stem + ext)
            if os.path.exists(candidate):
                vgdl_path = candidate
                break
        
        if vgdl_path is None:
            print(f"Nessun VGDL trovato per: {stem}")
            continue
        
        with open(desc_path, "r", encoding="utf-8") as f:
            description = f.read().strip()
        with open(vgdl_path, "r", encoding="utf-8") as f:
            vgdl_code = f.read().strip()
        
        pairs.append({"description": description, "vgdl": vgdl_code})
    
    return pairs

data = load_pairs()
print(f"Coppie caricate: {len(data)}")
dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
dataset.save_to_disk("dataset_hf")
print(dataset)
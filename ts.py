from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("input_smiles_100.csv")

# Convert SMILES to fingerprints
fps = []

for smi in df["SMILES"]:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fps.append(fp)

# -------------------------------
# Calculate pairwise similarity
# -------------------------------
similarities = []

for i in range(len(fps)):
    for j in range(i+1, len(fps)):
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        similarities.append(sim)

# -------------------------------
# Results
# -------------------------------
avg_sim = np.mean(similarities)
max_sim = np.max(similarities)
min_sim = np.min(similarities)

print("Average Similarity:", avg_sim)
print("Max Similarity:", max_sim)
print("Min Similarity:", min_sim)

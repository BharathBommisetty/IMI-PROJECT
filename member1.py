from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd
import numpy as np

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("input_smiles_100.csv")
df.columns = df.columns.str.strip().str.upper()

if "SMILES" not in df.columns:
    raise ValueError("SMILES column not found")

data = []

# -------------------------------
# Feature Extraction
# -------------------------------
for smi in df["SMILES"]:
    mol = Chem.MolFromSmiles(str(smi))

    if mol is None:
        print("❌ Invalid:", smi)
        data.append([np.nan]*11)
        continue

    try:
        # Basic descriptors
        mol_wt = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        mol_mr = Descriptors.MolMR(mol)

        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        ring_count = rdMolDescriptors.CalcNumRings(mol)

        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)

        # -------------------------------
        # YOUR ENGINEERED FEATURES
        # -------------------------------
        features = [
            mol_wt / num_atoms if num_atoms else 0,                 # ElasticModulus
            mol_wt / (tpsa + 1),                                    # Volume_Surface_Ratio
            num_atoms / (ring_count + 1),                           # Shape_Normalized
            rot_bonds / (num_bonds + 1),                            # Flexibility_Index
            mol_wt / (mol_mr + 1),                                  # Mass_Density
            tpsa / num_atoms if num_atoms else 0,                   # Polarity_Index
            logp / (h_donors + h_acceptors + 1),                    # Hydrophobicity_Balance
            mol_mr / (mol_wt + 1),                                  # Refractivity_Efficiency
            num_bonds / num_atoms if num_atoms else 0,              # Connectivity_Density
            ring_count / num_atoms if num_atoms else 0,             # Topological_Size_Ratio
            tpsa / (rot_bonds + 1)                                  # Surface_Stiffness_Ratio
        ]

    except Exception as e:
        print("⚠️ Error:", smi, "|", e)
        features = [np.nan]*11

    data.append(features)

# -------------------------------
# Column names
# -------------------------------
cols = [
    "ElasticModulus",
    "Volume_Surface_Ratio",
    "Shape_Normalized",
    "Flexibility_Index",
    "Mass_Density",
    "Polarity_Index",
    "Hydrophobicity_Balance",
    "Refractivity_Efficiency",
    "Connectivity_Density",
    "Topological_Size_Ratio",
    "Surface_Stiffness_Ratio"
]

# -------------------------------
# Save output
# -------------------------------
df_out = pd.concat([df, pd.DataFrame(data, columns=cols)], axis=1)

df_out.to_excel("ENGINEERED_FEATURES.xlsx", index=False)

print("✅ Engineered features extracted successfully!")

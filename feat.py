from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd

# -------------------------------
# Load your dataset
# -------------------------------
df = pd.read_csv("input_smiles_100.csv")   # Must contain a column named SMILES
df.columns = df.columns.str.strip().str.upper()

if "SMILES" not in df.columns:
    raise ValueError("SMILES column not found")

# -------------------------------
# Feature Extraction Function
# -------------------------------
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None]*30
    
    return [
        Descriptors.MolWt(mol),                      # 1 Molecular Weight
        Descriptors.MolLogP(mol),                    # 2 Hydrophobicity
        Descriptors.TPSA(mol),                       # 3 Polar Surface Area
        Descriptors.NumHDonors(mol),                 # 4 H-bond donors
        Descriptors.NumHAcceptors(mol),              # 5 H-bond acceptors
        Descriptors.NumRotatableBonds(mol),          # 6 Flexibility
        Descriptors.NumValenceElectrons(mol),        # 7 Valence electrons
        Descriptors.HeavyAtomCount(mol),             # 8 Heavy atoms
        Descriptors.RingCount(mol),                  # 9 Rings
        Descriptors.FractionCSP3(mol),               # 10 Saturation
        
        Descriptors.NumAliphaticRings(mol),          # 11
        Descriptors.NumAromaticRings(mol),           # 12
        Descriptors.NumSaturatedRings(mol),          # 13
        
        rdMolDescriptors.CalcNumAliphaticCarbocycles(mol), # 14
        rdMolDescriptors.CalcNumAromaticCarbocycles(mol),  # 15
        rdMolDescriptors.CalcNumSaturatedCarbocycles(mol), # 16
        
        rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),# 17
        rdMolDescriptors.CalcNumAromaticHeterocycles(mol), # 18
        rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),# 19
        
        Descriptors.NumRadicalElectrons(mol),        # 20
        Descriptors.MaxPartialCharge(mol),           # 21
        Descriptors.MinPartialCharge(mol),           # 22
        
        Descriptors.MaxAbsPartialCharge(mol),        # 23
        Descriptors.MinAbsPartialCharge(mol),        # 24
        
        Descriptors.BalabanJ(mol),                   # 25 Topological index
        Descriptors.BertzCT(mol),                    # 26 Complexity
        
        Descriptors.Chi0(mol),                       # 27 Connectivity index
        Descriptors.Chi1(mol),                       # 28
        Descriptors.Kappa1(mol),                     # 29 Shape index
        Descriptors.Kappa2(mol)                      # 30
    ]

# -------------------------------
# Apply feature extraction
# -------------------------------
features = df["SMILES"].apply(extract_features)

# Create DataFrame
feature_names = [
    "MolWt","LogP","TPSA","HDonors","HAcceptors","RotatableBonds",
    "ValenceElectrons","HeavyAtoms","RingCount","FractionCSP3",
    "AliphaticRings","AromaticRings","SaturatedRings",
    "AliphaticCarbocycles","AromaticCarbocycles","SaturatedCarbocycles",
    "AliphaticHeterocycles","AromaticHeterocycles","SaturatedHeterocycles",
    "RadicalElectrons","MaxPartialCharge","MinPartialCharge",
    "MaxAbsPartialCharge","MinAbsPartialCharge",
    "BalabanJ","BertzCT","Chi0","Chi1","Kappa1","Kappa2"
]

features_df = pd.DataFrame(features.tolist(), columns=feature_names)

# Combine with original data
final_df = pd.concat([df, features_df], axis=1)

# Save output
final_df.to_csv("polymer_features_30.csv", index=False)

print("✅ Feature extraction completed. Saved as polymer_features_30.csv")

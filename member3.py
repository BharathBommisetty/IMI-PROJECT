from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdPartialCharges
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("input_smiles_100.csv")
df.columns = df.columns.str.strip().str.upper()

data = []

for smi in df["SMILES"]:
    mol = Chem.MolFromSmiles(str(smi))

    if mol is None:
        print("❌ Invalid:", smi)
        data.append([np.nan]*10)
        continue

    try:
        # Compute charges
        rdPartialCharges.ComputeGasteigerCharges(mol)

        features = [
            Descriptors.NumValenceElectrons(mol),
            Descriptors.NumRadicalElectrons(mol),

            rdMolDescriptors.CalcNumSaturatedRings(mol),
            rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
            rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),

            rdMolDescriptors.CalcNumHeterocycles(mol),
            Descriptors.RingCount(mol),

            Descriptors.MaxAbsPartialCharge(mol),
            Descriptors.MinAbsPartialCharge(mol),

            Descriptors.ExactMolWt(mol)
        ]

    except Exception as e:
        print("⚠️ Error:", smi, "|", e)
        features = [np.nan]*10

    data.append(features)

# Column names
cols = [
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "NumSaturatedRings",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumHeterocycles",
    "RingCount",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "ExactMolWt"
]

# Save
df_out = pd.concat([df, pd.DataFrame(data, columns=cols)], axis=1)
df_out.to_excel("EXTRA_10_FEATURES.xlsx", index=False)

print("✅ Extra 10 features extracted!")

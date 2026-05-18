# ============================================================
# AI DRIVEN 3D PRINTABLE POLYMER DESIGN FRAMEWORK
# ============================================================

# ============================================================
# IMPORT MODULES
# ============================================================

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import joblib

RDLogger.DisableLog('rdApp.*')

# ============================================================
# USER INTERFACE
# ============================================================

print("\n" + "="*80)
print("AI DRIVEN 3D PRINTABLE POLYMER DESIGN FRAMEWORK")
print("="*80)

# ============================================================
# LOAD POLYMER DATASET
# ============================================================

print("\nLOADING POLYMER DATASET...\n")

dataset = pd.read_excel(
    "1000_Unique_Polymers_With_Names.xlsx"
)

print("DATASET LOADED SUCCESSFULLY")

print("TOTAL INPUT POLYMERS :", len(dataset))

# ============================================================
# CREATE POLYMER CHAINS
# ============================================================

print("\nCREATING POLYMER CHAINS...\n")

POLYMER_DEGREE = 3

polymer_names = []

polymer_smiles = []

for _, row in dataset.iterrows():

    name = str(row["Polymer_Name"])

    repeat_unit = str(row["Polymer_SMILES"])

    polymer_chain = repeat_unit * POLYMER_DEGREE

    polymer_names.append(name)

    polymer_smiles.append(polymer_chain)

print("POLYMER DEGREE :", POLYMER_DEGREE)

print("TOTAL POLYMER CHAINS :", len(polymer_smiles))

# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:

        atoms = mol.GetNumAtoms() or 1
        bonds = mol.GetNumBonds() or 1
        rings = rdMolDescriptors.CalcNumRings(mol) or 1

        return {

            "MolWt":
                Descriptors.MolWt(mol),

            "MolWt_TPSA_Ratio":
                Descriptors.MolWt(mol) /
                (Descriptors.TPSA(mol) + 1),

            "Atoms_Rings_Ratio":
                atoms / rings,

            "RotBond_Ratio":
                Descriptors.NumRotatableBonds(mol) /
                (bonds + 1),

            "MolWt_MolMR_Ratio":
                Descriptors.MolWt(mol) /
                (Descriptors.MolMR(mol) + 1),

            "TPSA_Atom_Ratio":
                Descriptors.TPSA(mol) / atoms,

            "LogP_HBond_Ratio":
                Descriptors.MolLogP(mol) /
                (
                    Descriptors.NumHDonors(mol) +
                    Descriptors.NumHAcceptors(mol) + 1
                ),

            "MolMR_MolWt_Ratio":
                Descriptors.MolMR(mol) /
                (Descriptors.MolWt(mol) + 1),

            "Bond_Atom_Ratio":
                bonds / atoms,

            "Ring_Atom_Ratio":
                rings / atoms,

            "ValenceElectrons":
                Descriptors.NumValenceElectrons(mol),

            "RadicalElectrons":
                Descriptors.NumRadicalElectrons(mol),

            "MaxAbsPartialCharge":
                Descriptors.MaxAbsPartialCharge(mol),

            "MinAbsPartialCharge":
                Descriptors.MinAbsPartialCharge(mol),

            "ExactMolWt":
                Descriptors.ExactMolWt(mol),

            "HeavyAtoms":
                mol.GetNumHeavyAtoms(),

            "Heteroatoms":
                Descriptors.NumHeteroatoms(mol),

            "FractionCSP3":
                Descriptors.FractionCSP3(mol),

            "AromaticRings":
                rdMolDescriptors.CalcNumAromaticRings(mol),

            "AliphaticRings":
                rdMolDescriptors.CalcNumAliphaticRings(mol),

            "Heterocycles":
                rdMolDescriptors.CalcNumHeterocycles(mol),

            "SaturatedRings":
                rdMolDescriptors.CalcNumSaturatedRings(mol),

            "ASA":
                rdMolDescriptors.CalcLabuteASA(mol),

            "MaxPartialCharge":
                Descriptors.MaxPartialCharge(mol),

            "MinPartialCharge":
                Descriptors.MinPartialCharge(mol),

            "HeavyAtomMolWt":
                Descriptors.HeavyAtomMolWt(mol),

            "NHOHCount":
                Descriptors.NHOHCount(mol),

            "NOCount":
                Descriptors.NOCount(mol)
        }

    except:

        return None

# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\nEXTRACTING FEATURES...\n")

feature_data = []

invalid_polymers = 0

for name, smiles in zip(polymer_names, polymer_smiles):

    feat = extract_features(smiles)

    if feat is not None:

        feat["Polymer_Name"] = name

        feat["Polymer_SMILES"] = smiles

        feature_data.append(feat)

    else:

        invalid_polymers += 1

df = pd.DataFrame(feature_data)

print("VALID POLYMERS :", len(df))

print("INVALID POLYMERS :", invalid_polymers)

# ============================================================
# MOLECULAR EMBEDDINGS
# ============================================================

print("\nGENERATING MOLECULAR EMBEDDINGS...\n")

fingerprints = []

embedding_smiles = []

for smi in df["Polymer_SMILES"]:

    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        continue

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=1024
    )

    arr = np.zeros((1024,), dtype=int)

    DataStructs.ConvertToNumpyArray(
        fp,
        arr
    )

    fingerprints.append(arr)

    embedding_smiles.append(smi)

fingerprints = np.array(fingerprints)

# ============================================================
# PCA REDUCTION
# ============================================================

pca = PCA(n_components=2)

embedding = pca.fit_transform(fingerprints)

# ============================================================
# SAVE EMBEDDINGS
# ============================================================

embedding_df = pd.DataFrame({

    "SMILES":
        embedding_smiles,

    "PCA1":
        embedding[:, 0],

    "PCA2":
        embedding[:, 1]
})

embedding_df.to_excel(
    "MOLECULAR_EMBEDDINGS.xlsx",
    index=False
)

print("MOLECULAR_EMBEDDINGS.xlsx SAVED")

# ============================================================
# EMBEDDING GRAPH
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(

    embedding[:, 0],
    embedding[:, 1],

    alpha=0.7
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("Molecular Embedding Space")

plt.savefig(
    "MOLECULAR_EMBEDDING_GRAPH.png",
    dpi=300
)

plt.close()

print("MOLECULAR_EMBEDDING_GRAPH.png SAVED")

# ============================================================
# TARGET CREATION
# ============================================================

def norm(x):

    return (
        x - x.min()
    ) / (
        x.max() - x.min() + 1e-6
    )

flexibility = norm(df["RotBond_Ratio"])

rigidity = norm(
    df["AromaticRings"] +
    df["AliphaticRings"]
)

adhesion = norm(df["TPSA_Atom_Ratio"])

size = norm(df["MolWt"])

size_score = 1 - abs(size - 0.5)

printability = (

    0.30 * (1 - flexibility) +
    0.30 * rigidity +
    0.20 * adhesion +
    0.20 * size_score
)

printability = np.clip(
    printability,
    0,
    1
)

df["PRINTABILITY"] = printability

# ============================================================
# SAVE DATASET
# ============================================================

df.to_excel(
    "PRINTABLE_POLYMER_DATASET.xlsx",
    index=False
)

print("PRINTABLE_POLYMER_DATASET.xlsx SAVED")

# ============================================================
# MACHINE LEARNING DATA
# ============================================================

X = df.drop(
    columns=[
        "Polymer_Name",
        "Polymer_SMILES",
        "PRINTABILITY"
    ]
)

y = df["PRINTABILITY"]

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y,

    test_size=0.2,

    random_state=42
)

# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(

    n_estimators=200,

    random_state=42
)

print("\nTRAINING RANDOM FOREST MODEL...\n")

model.fit(X_train, y_train)

# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(X_test)

# ============================================================
# MODEL METRICS
# ============================================================

r2 = r2_score(
    y_test,
    predictions
)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

print("\n" + "="*80)
print("FORWARD MODEL PERFORMANCE")
print("="*80)

print("R² SCORE :", round(r2, 4))

print("MAE :", round(mae, 4))

print("RMSE :", round(rmse, 4))

# ============================================================
# FORWARD PLOT
# ============================================================

plt.figure(figsize=(7,7))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual Printability")

plt.ylabel("Predicted Printability")

plt.title("Actual vs Predicted")

plt.savefig(
    "FORWARD_ACTUAL_VS_PREDICTED.png"
)

plt.close()

print("FORWARD_ACTUAL_VS_PREDICTED.png SAVED")

# ============================================================
# ERROR DISTRIBUTION
# ============================================================

errors = np.abs(
    np.array(y_test) -
    np.array(predictions)
)

plt.figure(figsize=(8,5))

plt.hist(
    errors,
    bins=25
)

plt.xlabel("Prediction Error")

plt.ylabel("Frequency")

plt.title("Error Distribution")

plt.savefig(
    "FORWARD_ERROR_DISTRIBUTION.png"
)

plt.close()

print("FORWARD_ERROR_DISTRIBUTION.png SAVED")

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "printability_model.pkl"
)

print("printability_model.pkl SAVED")

# ============================================================
# PREDICT PRINTABILITY
# ============================================================

def predict_printability(smiles):

    polymer_chain = smiles * POLYMER_DEGREE

    feat = extract_features(polymer_chain)

    if feat is None:

        print("\nINVALID POLYMER")

        return

    feat_df = pd.DataFrame([feat])

    feat_scaled = scaler.transform(feat_df)

    pred = model.predict(feat_scaled)[0]

    print("\nRESULT")

    print("POLYMER CHAIN :", polymer_chain)

    print("PREDICTED PRINTABILITY :", round(pred, 4))

# ============================================================
# REAL POLYMER INVERSE DESIGN
# ============================================================

def inverse_design(

    target=0.8,

    top_n=5
):

    print("\nSEARCHING REAL POLYMERS...\n")

    predictions = model.predict(X_scaled)

    results_df = pd.DataFrame({

        "Polymer_Name":
            df["Polymer_Name"],

        "Polymer_SMILES":
            df["Polymer_SMILES"],

        "Predicted_Printability":
            predictions
    })

    results_df["Error"] = abs(

        results_df["Predicted_Printability"] -

        target
    )

    results_df = results_df.sort_values(

        by="Error"
    )

    top_polymers = results_df.head(top_n)

    print("\nTOP MATCHING REAL POLYMERS\n")

    print(
        top_polymers.to_string(index=False)
    )

    top_polymers.to_excel(

        "TOP_REAL_PRINTABLE_POLYMERS.xlsx",

        index=False
    )

    print(
        "\nTOP_REAL_PRINTABLE_POLYMERS.xlsx SAVED"
    )

    # ========================================================
    # TARGET VS POLYMERS GRAPH
    # ========================================================

    plt.figure(figsize=(8,5))

    plt.bar(

        range(len(top_polymers)),

        top_polymers["Predicted_Printability"]
    )

    plt.axhline(

        y=target,

        linestyle='--'
    )

    plt.xticks(

        range(len(top_polymers)),

        top_polymers["Polymer_Name"],

        rotation=15
    )

    plt.ylabel("Printability")

    plt.title("Target vs Real Polymers")

    plt.tight_layout()

    plt.savefig(

        "TARGET_VS_REAL_POLYMERS.png"
    )

    plt.close()

    print(
        "TARGET_VS_REAL_POLYMERS.png SAVED"
    )

# ============================================================
# USER INTERFACE
# ============================================================

while True:

    print("\n1. Predict Printability")

    print("2. Real Polymer Inverse Design")

    print("3. Exit")

    choice = input("\nEnter Choice : ")

    if choice == "1":

        smi = input("\nEnter Polymer Repeat Unit SMILES : ")

        predict_printability(smi)

    elif choice == "2":

        try:

            target = float(

                input(
                    "\nEnter Target Printability (0-1) : "
                )
            )

            inverse_design(target)

        except:

            print("\nINVALID INPUT")

    elif choice == "3":

        print("\nPROGRAM FINISHED")

        break

    else:

        print("\nINVALID CHOICE")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*80)
print("FINAL PROJECT SUMMARY")
print("="*80)

print("TOTAL INPUT POLYMERS :", len(dataset))

print("VALID POLYMERS :", len(df))

print("INVALID POLYMERS :", invalid_polymers)

print("\nMODEL PERFORMANCE")

print("R² :", round(r2, 4))

print("MAE :", round(mae, 4))

print("RMSE :", round(rmse, 4))

print("\nOUTPUT FILES")

print("1. PRINTABLE_POLYMER_DATASET.xlsx")

print("2. MOLECULAR_EMBEDDINGS.xlsx")

print("3. printability_model.pkl")

print("4. MOLECULAR_EMBEDDING_GRAPH.png")

print("5. FORWARD_ACTUAL_VS_PREDICTED.png")

print("6. FORWARD_ERROR_DISTRIBUTION.png")

print("7. TOP_REAL_PRINTABLE_POLYMERS.xlsx")

print("8. TARGET_VS_REAL_POLYMERS.png")

print("\nFRAMEWORK COMPLETED SUCCESSFULLY")

print("="*80)

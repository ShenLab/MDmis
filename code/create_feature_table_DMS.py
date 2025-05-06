import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import glob
from utils import *

results_dir = "/home/az2798/MDmis/results/"
vault_dir = "/share/vault/Users/az2798/"
data_dir = "/home/az2798/MDmis/data/"

DMS_labels_df = pd.read_csv(os.path.join(vault_dir, "DMS_Data", "DMS_labels.csv")) 
MD_metadata = pd.read_csv(os.path.join(data_dir, "MD_metadata.csv"),
                                    index_col=0)


MD_metadata.rename(columns={"source": "MD Data Source"}, inplace=True)
IDRome_metadata = MD_metadata[MD_metadata["MD Data Source"] == "IDRome"]

print(DMS_labels_df["UniProtID"].nunique())




### The first merge between MD_metadata and DMS Labels

DMS_proteins = pd.merge(left = IDRome_metadata,
                          right = DMS_labels_df, on = "UniProtID", how = "inner")


# #print(DMS_proteins.head())
# print(DMS_proteins[DMS_proteins["UniProtID"]=="Q13148"])
# print(DMS_proteins["UniProtID"].nunique(), "After Merging")
# print(DMS_proteins["UniProtID"].unique(), "After Merging")


DMS_proteins["start"] = DMS_proteins["protein_start_end"].str.split("_").str[1].astype(int)
DMS_proteins["end"] = DMS_proteins["protein_start_end"].str.split("_").str[2].astype(int)

subset_mapped_proteins = DMS_proteins[
    (DMS_proteins["Location"] >= DMS_proteins["start"]) &
    (DMS_proteins["Location"] <= DMS_proteins["end"])
].drop_duplicates(subset = ["UniProtID", "start", "end", "Location"])
subset_mapped_proteins.to_csv(f'{data_dir}MDmis_DMS.csv')

print(subset_mapped_proteins.head())
print(subset_mapped_proteins["protein_start_end"].unique())

###
#Creating a damage score (as a rank)

# subset_mapped_proteins["Normalized_Damage_Rank"] = (
#     subset_mapped_proteins.groupby("UniProtID")["DMS_score"]
#     .transform(lambda x: x.mul(-1).rank() / len(x))
# )

### Now creating a feature table with MD data
# Remove non-IDRs and HGMD labels
h5py_path = "/share/vault/Users/az2798/train_data_all/filtered_feature_all_ATLAS_GPCRmd_IDRome.h5"

aa_order = "ARNDCQEGHILKMFPSTWYV"
aaindex_res_mat = np.load(os.path.join(data_dir, "aa_index1.npy"))
res_data, pair_data = load_MD_data(h5py_path)


IDRome_labels = subset_mapped_proteins[(subset_mapped_proteins["MD Data Source"] == "IDRome") &
                                       (subset_mapped_proteins["Changed_Residue"] != "X")]

unique_proteins = IDRome_labels["UniProtID"].unique()
print("Unique proteins", len(unique_proteins))

unique_proteins_df = pd.DataFrame(unique_proteins, columns=["UniProtID"])


variant_information_columns = ["DMS_z_score", "UniProtID",
                               "protein_start_end",
                                "Location", "Original AA", 
                               "Changed AA", "Average SASA",
                               "Average RMSF"]
location_column_name = "Location"
original_aa_column_name = "Original_AA"
changed_aa_column_name = "Changed_Residue" 
outcome_column_name = "DMS_z_score"

# ESM_dir = "/share/vault/Users/gz2294/Data/DMS/ClinVar.HGMD.PrimateAI.syn/esm2.650M.embedding.uniprotIDs/"
# ESM_data = {}

# for file in glob.glob(os.path.join(ESM_dir, "*representations*")):
#     uniprot_ID = os.path.basename(file).split(".")[0]
#     ESM_data[uniprot_ID] = np.load(file)

entire_feature_table = create_feature_table(IDRome_labels, res_data, pair_data,
                                        aaindex_res_mat,
                                        aa_order,3,
                                        location_column_name,
                                        None, # label source is irrelevant
                                        original_aa_column_name,
                                        changed_aa_column_name, 
                                        outcome_column_name
                                        )
print(entire_feature_table.shape)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#Iterate through each fold
for fold_idx, (train_index, val_index) in enumerate(kf.split(unique_proteins_df)):
    train_proteins = unique_proteins_df.iloc[train_index]
    val_proteins = unique_proteins_df.iloc[val_index]

    train_proteins_df = IDRome_labels[IDRome_labels["UniProtID"].isin(train_proteins["UniProtID"])]
    val_proteins_df = IDRome_labels[IDRome_labels["UniProtID"].isin(val_proteins["UniProtID"])]
    

    print(train_proteins_df.shape)
    
    train_feature_table = entire_feature_table[entire_feature_table["UniProtID"].isin(train_proteins["UniProtID"])]
    val_feature_table = entire_feature_table[entire_feature_table["UniProtID"].isin(val_proteins["UniProtID"])]

    fold_dir = os.path.join(data_dir, "DMS_train_val", f"fold_{fold_idx + 1}")
    os.makedirs(fold_dir, exist_ok=True)
    train_feature_table.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
    val_feature_table.to_csv(os.path.join(fold_dir, "val.csv"), index=False)

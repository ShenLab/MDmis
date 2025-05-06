import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from utils import *
import glob

results_dir = "/home/az2798/MDmis/results/"
vault_dir = "/share/vault/Users/az2798/"
data_dir = "/home/az2798/MDmis/data/"

clinvar_labels_df = pd.read_csv(os.path.join(vault_dir, "ClinVar_Data", "training.csv")) 
MD_metadata = pd.read_csv(os.path.join(data_dir, "MD_metadata.csv"),
                                    index_col=0)

MD_metadata.rename(columns={"source": "MD Data Source"}, inplace=True)
IDRome_metadata = MD_metadata[MD_metadata["MD Data Source"] == "IDRome"]
clinvar_labels_df.rename(columns = {"score":"outcome", "uniprotID":"UniProtID", "pos.orig": "location", 
                       "alt": "changed_residue", 'data_source': 'Label Source',
                       "Ensembl_transcriptid": "ENST_id"}, inplace=True)
clinvar_labels_df = clinvar_labels_df[["outcome", "UniProtID", "location", "changed_residue", "VarID", 
                          "Label Source", "ENST_id"]]

### Clinvar Labels Df is our background
print(clinvar_labels_df.shape)
### The first merge between MD_metadata and Clinvar Labels

clinvar_proteins = pd.merge(left = IDRome_metadata,
                          right = clinvar_labels_df, on = "UniProtID", how = "inner")


print(clinvar_proteins.head())


clinvar_proteins["location"] = clinvar_proteins["location"].astype(int)
clinvar_proteins["start"] = clinvar_proteins["protein_start_end"].str.split("_").str[1].astype(int)
clinvar_proteins["end"] = clinvar_proteins["protein_start_end"].str.split("_").str[2].astype(int)

subset_mapped_proteins = clinvar_proteins[
    (clinvar_proteins["location"] >= clinvar_proteins["start"]) &
    (clinvar_proteins["location"] <= clinvar_proteins["end"])
].drop_duplicates(subset = ["UniProtID", "start", "end", "location"])
subset_mapped_proteins.to_csv(f'{data_dir}MDmis_clinical.csv')

#Removing potential start loss missense variants
print(subset_mapped_proteins.shape)
subset_mapped_proteins = subset_mapped_proteins[subset_mapped_proteins["location"] != 1]


print(subset_mapped_proteins.shape)
### Now creating a feature table with MD data
# Remove non-IDRs and HGMD labels


h5py_path = "/share/vault/Users/az2798/train_data_all/filtered_feature_all_ATLAS_GPCRmd_IDRome.h5"

aa_order = "ARNDCQEGHILKMFPSTWYV"
aaindex_res_mat = np.load(os.path.join(data_dir, "aa_index1.npy"))
res_data, pair_data = load_MD_data(h5py_path)


IDRome_labels = subset_mapped_proteins[subset_mapped_proteins["changed_residue"] != "X"]

unique_proteins = IDRome_labels["UniProtID"].unique()
print("Unique protein regions", len(IDRome_labels["protein_start_end"].unique()))
print("Unique Proteins", len(unique_proteins))
print(IDRome_labels.shape)

unique_proteins_df = pd.DataFrame(unique_proteins, columns=["UniProtID"])

####
print(IDRome_labels[["outcome", "Label Source"]].value_counts())

# # train_proteins, val_proteins = train_test_split(unique_proteins_df, test_size=0.2, random_state=42)
# ESM_dir = "/share/vault/Users/gz2294/Data/DMS/ClinVar.HGMD.PrimateAI.syn/esm2.650M.embedding.uniprotIDs/"
# ESM_data = {}

# for file in glob.glob(os.path.join(ESM_dir, "*representations*")):
#     uniprot_ID = os.path.basename(file).split(".")[0]
#     ESM_data[uniprot_ID] = np.load(file)

# print(len(ESM_data.keys()), "Number of embeddings")



location_column_name = "location"
label_source_column_name = "Label Source"
original_aa_column_name = None
changed_aa_column_name = "changed_residue" 
outcome_column_name = "outcome"

entire_feature_table = create_feature_table(
        IDRome_labels, res_data, pair_data,
        aaindex_res_mat,
        aa_order, 3, 
        location_column_name,
        label_source_column_name,
        original_aa_column_name, changed_aa_column_name, outcome_column_name
    )

entire_feature_table.to_csv(os.path.join(data_dir, "clinical_train_val", "feature_table.csv"), index=False)

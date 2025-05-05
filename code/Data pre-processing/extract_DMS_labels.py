import numpy as np
import pickle
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import scipy.stats as ss

import glob
import sys
sys.path.append('/home/az2798/MDmis/code/')


from utils import *
from process_proteome_information import *

def parse_DMS_binary(UniProt_ID, DMS_path, DMS_directionality):
    DMS_df = pd.read_csv(DMS_path) 
    DMS_df = DMS_df[~DMS_df["mutant"].str.contains(":")]
    DMS_df[['Original_AA', 'Location', 'Changed_Residue']] = DMS_df['mutant'].str.extract(r'([A-Z])(\d+)([A-Z])')

    DMS_df["Location"] = DMS_df["Location"].astype(int)

    DMS_df["UniProtID"] = UniProt_ID

    DMS_df["VarID"] = DMS_df["UniProtID"] + ":" + DMS_df["mutant"]
    
    
    DMS_df["DMS_score"] = DMS_df["DMS_score"]* -1 * DMS_directionality #if directionality is 1, indicates "fitness", not damage

    DMS_df["DMS_z_score"] = ss.zscore(DMS_df["DMS_score"])

    DMS_df["DMS_score_bin"] = DMS_df["DMS_score_bin"].map({0: 1, 1: 0})

    return DMS_df

def create_DMS_table(DMS_metadata, DMS_dir):
    list_of_DMS_files = glob.glob(os.path.join(DMS_dir, '*'))
    
    human_DMS_files = [file for file in list_of_DMS_files if 'HUMAN' in os.path.basename(file)]

    df_list = []
    
    for idx, row in DMS_metadata.iterrows():
        file_name = row['File_Name']
        uniprot_id = row['UniProtID']
        assay_type = row['Assay_Type']
        directionality = row['DMS_directionality']
        matching_files = [file for file in human_DMS_files if file_name in os.path.basename(file)]
        
        for DMS_path in matching_files:
            print(DMS_path)
            DMS_df = parse_DMS_binary(uniprot_id, DMS_path,directionality)
            DMS_df["File_Name"] = os.path.basename(DMS_path)
            DMS_df["Assay_Type"] = assay_type
            
            df_list.append(DMS_df)

    combined_DMS_table = pd.concat(df_list, ignore_index=True)

    conflict_mask = combined_DMS_table.groupby(
            ['UniProtID', 'Location', 'Changed_Residue']
        )['DMS_score_bin'].transform(lambda x: x.nunique() > 1)

    combined_DMS_table = combined_DMS_table[~conflict_mask]
    
    print(combined_DMS_table.shape)
    print(combined_DMS_table.head())

    return combined_DMS_table

def main():
    data_dir = "/home/az2798/MDmis/data/"
    vault_dir = "/share/vault/Users/az2798/"
    results_dir = "/home/az2798/MDmis/results/"
    DMS_metadata = pd.read_csv(
        os.path.join(data_dir, "DMS_metadata.txt"), sep= r'\s+'
    )

    print(DMS_metadata)

    DMS_labels_df = create_DMS_table(DMS_metadata, os.path.join(data_dir, "DMS_ProteinGym_substitutions"))
    
    DMS_labels_df = DMS_labels_df[DMS_labels_df["Location"] !=1] #removing start loss variants

    # store for long term
    DMS_labels_df.to_csv(os.path.join(
        vault_dir, "DMS_Data", "DMS_labels.csv"
    ))
    print(DMS_labels_df["UniProtID"].nunique())

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=DMS_labels_df, 
        x="DMS_z_score", 
        hue="UniProtID", 
        kde=False, 
        palette="viridis", 
        bins=30
    )
    plt.title("Histogram of DMS Scores Colored by UniProtID")
    plt.xlabel("DMS z Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,
        "DMS_scores_by_protein.png"), dpi=300
    )

    plt.clf()
    g = sns.FacetGrid(DMS_labels_df, col="DMS_score_bin", col_wrap=3, height=4, sharex=True, sharey=True)
    g.map(sns.histplot, "DMS_z_score", bins=30, color="blue", kde=False)
    g.set_titles("{col_name}")
    g.set_axis_labels("DMS z Score", "Count")
    g.figure.suptitle("Histogram of DMS Scores by DMS Score Bin", y=1.05)
    g.tight_layout()
    plt.savefig(os.path.join(results_dir,
        "DMS_scores_by_label.png"), dpi=300
    )
    plt.clf()
    
    plt.clf()
    g = sns.FacetGrid(DMS_labels_df, col="Assay_Type", col_wrap=3, height=4, sharex=True, sharey=False)
    g.map(sns.histplot, "DMS_score", bins=30, color="blue", kde=False)
    g.set_titles("{col_name}")
    g.set_axis_labels("DMS Score", "Count")
    g.figure.suptitle("Histogram of DMS Scores by Assay Type", y=1.05)
    g.tight_layout()
    plt.savefig(os.path.join(results_dir,
        "DMS_scores_by_assay.png"), dpi=300
    )
    plt.clf()

    # ### Load the GERP data

    # GERP_df = pd.read_csv(os.path.join(
    #     vault_dir, "dbNSFP_data", "dbNSFP4_merged.csv"
    #     ), index_col=0, header=0)
    # GERP_columns_to_split = ['aapos', 'Ensembl_transcriptid', 'Uniprot_acc']
    # GERP_char_to_split = ";"

    # ### Load Amis Table
    # alpha_missense_table = pd.read_csv(os.path.join(data_dir,
    #                             "AlphaMissense_data", "AlphaMissense_aa_substitutions.tsv"),
    #                             sep='\t', low_memory=False, skiprows=[0,1,2])
    # ###
    # pLDDT_df = pd.read_csv(os.path.join(data_dir, 
    #                                         "pLDDT_scores", 
    #                                         "processed_pLDDT_scores.csv"), 
    #                             index_col=0, low_memory=False)
    
    # ### Preprocess and concatenate MSA coevolution features
    # list_of_MSA_feature_paths = glob.glob(os.path.join(data_dir, 
    #     "MSA_coevolution_features_MI",
    #     "MSA_coevolution_features_*"))
    # list_of_MSA_feature_dfs = [pd.read_csv(path, index_col = 0) for path in list_of_MSA_feature_paths]

    # MSA_feature_table_combined = pd.concat(list_of_MSA_feature_dfs, axis = 0)
    # MSA_feature_table_combined.rename({"ENST_id.1": "Gene_id"}, axis=1, inplace=True)
    # MSA_feature_table_combined = MSA_feature_table_combined[MSA_feature_table_combined["Location_Sequence"]!= "-"] #removing gaps
    # MSA_feature_table_combined["Location_Sequence"] = MSA_feature_table_combined["Location_Sequence"].astype(int)
    # ###

    # merged_proteome_table_DMS = preprocess_proteome_information(DMS_labels_df, 
    #                                 "Location", "Changed_Residue",
    #                                 GERP_df, GERP_columns_to_split, GERP_char_to_split,
    #                                 alpha_missense_table, ESM pLDDT_df,
    #                                 MSA_feature_table_combined
    #                                 )
    

    # merged_proteome_table_DMS.to_csv(os.path.join(data_dir,
    #     "merged_proteome_information_DMS.csv"))
if __name__ == "__main__":
    main()

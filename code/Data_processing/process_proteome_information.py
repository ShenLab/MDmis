
import os
import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import glob
from utils import *
import argparse


def preprocess_proteome_information(labels_df,
        labels_location_column_name, labels_changed_aa_column_name,
        GERP_df, GERP_columns_to_split, GERP_char_to_split,
        alpha_missense_table, ESM_table,
        pLDDT_df, MSA_coevolution_df):


    print(len(GERP_df["Uniprot_acc"].unique()), "Unique Proteins in GERP")
    GERP_df_split = split_rows(GERP_df, GERP_columns_to_split, GERP_char_to_split)


    ### Preprocess the AlphaMissense Table


    alpha_missense_table = alpha_missense_table[alpha_missense_table["uniprot_id"].isin(
        labels_df["UniProtID"].unique()
        )]

    alpha_missense_table['location_amis'] = alpha_missense_table['protein_variant'].str.extract(r'(\d+)')

    alpha_missense_table['location_amis'] = alpha_missense_table['location_amis'].astype(int)
    alpha_missense_table['changed_aa_amis'] = alpha_missense_table['protein_variant'].str.extract(r'([A-Z])$', expand=False)

    ### Merge Labels with Amis and GERP
    print(labels_df.shape, "Labels shape")
    amis_labels_merged = pd.merge(left=labels_df, right=alpha_missense_table,
            left_on= ["UniProtID", labels_location_column_name,
                       labels_changed_aa_column_name],
            right_on= ["uniprot_id", "location_amis", "changed_aa_amis"],
            how="left", indicator=True)
    
    print(amis_labels_merged["_merge"].value_counts(), "Check 1")

    amis_labels_merged.drop(columns=["_merge"], inplace = True)

    ### Merge Amis & GERP
    GERP_df_split["aapos"] = GERP_df_split["aapos"].astype(int)
    amis_GERP_merged = pd.merge(left=amis_labels_merged, 
                         right=GERP_df_split, 
                         left_on= ["UniProtID", labels_location_column_name],
                         right_on=["Uniprot_acc", "aapos"],
                         how="left", 
                         indicator=True)
    
    print(amis_GERP_merged.shape, "After Amis + Gerp shape")
    #print(amis_GERP_merged["_merge"].value_counts(), "Check 2")

    amis_GERP_merged.drop_duplicates(subset= ["UniProtID",
                                              labels_location_column_name,
                                              labels_changed_aa_column_name], inplace=True)
    
    #print(amis_GERP_merged["_merge"].value_counts(), "Check 2.5")

    print(amis_GERP_merged.shape, "Amis Merging Shape")
    print(amis_GERP_merged["am_pathogenicity"].isnull().sum())
    print(amis_GERP_merged[amis_GERP_merged["am_pathogenicity"].isna()])

    amis_GERP_merged.drop(columns=["_merge"], inplace = True)


    ### Merge with ESM1b scores
    print(amis_GERP_merged.shape, "Before ESM")
    merged_ESM = pd.merge(left =amis_GERP_merged, right = ESM_table,
                          left_on= ["UniProtID", labels_location_column_name, labels_changed_aa_column_name],
                          right_on=["Protein_ID", "Pos", "Changed_AA"], 
                          how = "left",
                          suffixes=["", "_y"])
    print(merged_ESM.shape, "After ESM")

    ### Merge with pLDDT and co-evolution

    merged_pLDDT = pd.merge(left = merged_ESM,
                            right = pLDDT_df,
             left_on=["UniProtID", labels_location_column_name],
             right_on=["UniProtID", "location"],
             how = "left", indicator=True)

    merged_pLDDT.drop(columns=["_merge"], inplace = True)

    ### Merge with co-evolutionary signal


    merged_coevolution = pd.merge(left = merged_pLDDT,
                                right = MSA_coevolution_df,
             left_on=["Ensembl_transcriptid", labels_location_column_name],
             right_on=["ENST_id", "Location_Sequence"],
             how = "left", indicator=True)

    
    merged_coevolution.drop(columns=["_merge"], inplace = True)

    return merged_coevolution



def main():

    ### Define paths

    data_dir = "/home/az2798/MDmis/data/"
    vault_dir = "/nfs/user/Users/az2798/"

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clinical", action="store_true",
                        help="If set, the data processing will be done on clinical labels. Otherwise, it will do the DMS labels.")
    args = parser.parse_args()
    clinical = args.clinical
    print("Processing clinical information", clinical)
    if clinical: ## does it for clinical labels, ClinVar + PrimateAI primarily
        clinvar_labels_df = pd.read_csv(os.path.join(vault_dir, "ClinVar_Data", "training.csv")) 
        clinvar_labels_df.rename(columns = {"score":"outcome", "uniprotID":"UniProtID", "pos.orig": "location", 
                            "alt": "changed_residue", 'data_source': 'Label Source'}, inplace=True)
        clinvar_labels_df = clinvar_labels_df[["outcome", "UniProtID", "location", "changed_residue", "VarID", 
                                "Label Source"]]
        
        processed_labels_df = clinvar_labels_df[clinvar_labels_df["location"] !=1] #removing start loss variants
        location_column_name = "location"
        changed_residue_column = "changed_residue" #FIXME
    else:
        processed_labels_df = pd.read_csv(os.path.join(
        vault_dir, "DMS_Data", "DMS_labels.csv"
        ))
        location_column_name = "Location"
        changed_residue_column = "Changed_Residue"

    #Same process as done in feature table generation

    ### Load the GERP data

    GERP_df = pd.read_csv(os.path.join(
        vault_dir, "dbNSFP_data", "dbNSFP4_merged.csv"
        ), index_col=0, header=0)
    GERP_columns_to_split = ['aapos', 'Ensembl_transcriptid', 'Uniprot_acc']
    GERP_char_to_split = ";"
    ### Load Amis Table
    alpha_missense_table = pd.read_csv(os.path.join(data_dir,
                                "AlphaMissense_data", "AlphaMissense_aa_substitutions.tsv"),
                                sep='\t', low_memory=False, skiprows=[0,1,2])
    
    ### Load ESM1b LLRs
    ESM_table = pd.read_csv(os.path.join(vault_dir,
                                "ESM1b_data", "All_LLR_scores.csv"),
                                low_memory=False, index_col=0)
    scaler = MinMaxScaler()
    ESM_table["ESM_probabilities"] = scaler.fit_transform(-1*ESM_table[['LLR']])
    ###
    pLDDT_df = pd.read_csv(os.path.join(data_dir, 
                                            "pLDDT_scores", 
                                            "processed_pLDDT_scores.csv"), 
                                index_col=0, low_memory=False)
    
    ### Preprocess and concatenate MSA coevolution features
    list_of_MSA_feature_paths = glob.glob(os.path.join(data_dir, 
        "MSA_coevolution_features_MI",
        "MSA_coevolution_features_*"))
    list_of_MSA_feature_dfs = [pd.read_csv(path, index_col = 0) for path in list_of_MSA_feature_paths]

    MSA_feature_table_combined = pd.concat(list_of_MSA_feature_dfs, axis = 0)
    MSA_feature_table_combined.rename({"ENST_id.1": "Gene_id"}, axis=1, inplace=True)
    MSA_feature_table_combined = MSA_feature_table_combined[MSA_feature_table_combined["Location_Sequence"]!= "-"] #removing gaps
    MSA_feature_table_combined["Location_Sequence"] = MSA_feature_table_combined["Location_Sequence"].astype(int)
    ###

    merged_proteome_information = preprocess_proteome_information(processed_labels_df, 
                                    location_column_name, changed_residue_column,
                                    GERP_df, GERP_columns_to_split, GERP_char_to_split,
                                    alpha_missense_table, ESM_table, pLDDT_df,
                                    MSA_feature_table_combined
                                    )
    
    if clinical:
        merged_proteome_information.to_csv(os.path.join(data_dir,
        "merged_proteome_information_clinical.csv"))
    else:
        merged_proteome_information.to_csv(os.path.join(data_dir,
        "merged_proteome_information_DMS.csv"))

if __name__ == "__main__":
    main()
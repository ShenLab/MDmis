import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import sys
import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.append(ROOT)
from utils import *
from config import config



def create_feature_table_GPCRmd(df_table, res_data, pair_data, aa_res_matrix,
                        aa_order, variant_information_columns,
                        location_column_name = None, 
                        original_aa_column_name = None,
                        changed_aa_column_name = None,
                        outcome_column_name = None,
                        use_res_md = True, use_pair_md = True):
    """
    Creates a feature table combining residue-level and pairwise molecular dynamics (MD) data,
    amino acid (AA) features, and additional variant-specific information for each protein variant.

    Parameters:
    ----------
    df_table : pandas.DataFrame
        A DataFrame containing information about protein variants, including location, sequence, 
        and metadata like "outcome", "location", etc.
        
    res_data : dict
        A dictionary mapping protein regions ("protein_start_end") to corresponding residue-level MD features.
        
    pair_data : dict
        A dictionary mapping protein regions ("protein_start_end") to corresponding pairwise MD features.
        
    aa_res_matrix : numpy.ndarray
        A matrix that represents amino acid-specific features, where rows correspond to amino acids, 
        and columns correspond to their respective feature vectors.
        
    aa_order : list
        A list of amino acids in the same order as the indices of `aa_res_matrix`, used to map amino acids 
        to feature vectors.
        
    variant_information_columns : list
        A list of column names corresponding to the variant-specific information to be included in the feature table 
        (e.g., "outcome", "location", etc.).
        
    location_column_name : str
        The column name that should be used to access the variant's location in the sequence

    original_aa_column_name : str
        The column name that should be used to access the original AA in the sequence
    
    changed_aa_column_name : str
        The column name that should be used to access the changed AA in the sequence
    
    outcome_column_name : str
        The column name that should be used to access the outcome label 


    use_res_md : bool, optional
        A flag indicating whether to include residue-level MD features. Default is True.
        
    use_pair_md : bool, optional
        A flag indicating whether to include pairwise MD features. Default is True.

    Returns:
    --------
    feature_table : pandas.DataFrame
        A DataFrame containing the extracted features for each variant, including amino acid features, 
        residue-level MD features, pairwise MD features (if applicable), and additional variant-specific information.

    Notes:
    ------
    - Residue-level and pairwise MD features are extracted based on the location of the variant in the sequence.
    - If MD data for a protein is missing in either `res_data` or `pair_data`, the variant is skipped.
    - The final feature table combines AA features, MD features (if applicable), and variant-specific metadata.
    """

    aa_to_index = {aa: idx for idx, aa in enumerate(aa_order)}

    res_md_columns = [f'Res_MD_{i+1}' for i in range(47)] # number of Res MD features
    pair_md_columns = [f'Avg_Pair_MD_{i+1}' for i in range(10)] # number of Pair MD features
    aa_columns = [f'Res_AA_{i+1}' for i in range(aa_res_matrix.shape[1])]
    #column_names = md_columns + aa_columns + ['outcome']
    column_names = aa_columns

    if use_res_md:
        column_names = column_names + res_md_columns
    if use_pair_md:
        column_names = column_names + pair_md_columns

    column_names = column_names + variant_information_columns

    feature_table = pd.DataFrame(columns=column_names)
    
    
    for index, row in df_table.iterrows():
        #query res_data
        location_variant = row[location_column_name]
        index_protein_sequence = location_variant - int(row["sstart"]) 


        print(len(row["sequence"]))
        print(res_data[row["MD_protein_start_end"]][row["MD_start"]-1:row["MD_end"]].shape)
        if row["MD_protein_start_end"] not in res_data or row["MD_protein_start_end"] not in pair_data:
            continue
        residue_md_features = np.array(res_data[row["MD_protein_start_end"]][ row["MD_start"]-1:row["MD_end"] ][index_protein_sequence])
        
        avg_pair_md_features = np.mean(pair_data[row["MD_protein_start_end"]][ row["MD_start"]-1:row["MD_end"] ][index_protein_sequence, :, :], axis = 0)

        if original_aa_column_name == None:
            original_aa = row["sequence"][index_protein_sequence] # string
        else:
            original_aa = row[original_aa_column_name]
        new_aa = row[changed_aa_column_name] # string
        
        residue_aa_index_features = aa_res_matrix[aa_to_index[new_aa], :] - aa_res_matrix[aa_to_index[original_aa]]

        variant_information = row[[outcome_column_name, "UniProtID", location_column_name,
                                   "MD_protein_start_end"]].values
        variant_information = np.concatenate(
            (variant_information, np.array([original_aa, new_aa]) )
        )
        data_row = np.array(residue_aa_index_features)
        if use_res_md:
            data_row = np.concatenate((data_row, residue_md_features), axis=None)

        if use_pair_md:
            data_row = np.concatenate((data_row, avg_pair_md_features), axis=None)
       
        feature_table.loc[len(feature_table.index)] = np.concatenate((data_row, variant_information),
                                                                        axis=None)
  
    return feature_table

def main():
    vault_dir = os.path.abspath(config["vault_dir"])
    data_dir = os.path.abspath(config["data_dir"])
    clinvar_labels_df = pd.read_csv(os.path.join(vault_dir, "ClinVar_Data", "training.csv")) 
    GPCRmd_metadata = pd.read_csv(os.path.join(data_dir, "GPCRmd_metadata_processed.csv"),
                                        index_col=0)

    GPCRmd_metadata.rename(columns={"source": "MD Data Source"}, inplace=True)

    clinvar_labels_df.rename(columns = {"score":"outcome", "uniprotID":"UniProtID", "pos.orig": "location", 
                        "alt": "changed_residue", 'data_source': 'Label Source',
                        "Ensembl_transcriptid": "ENST_id"}, inplace=True)
    clinvar_labels_df = clinvar_labels_df[["outcome", "UniProtID", "location", "changed_residue", "VarID", 
                            "Label Source", "ENST_id"]]

    ### Clinvar Labels Df is our background
    print(clinvar_labels_df.shape)
    ### The first merge between MD_metadata and Clinvar Labels

    clinvar_proteins = pd.merge(left = GPCRmd_metadata,
                            right = clinvar_labels_df, on = "UniProtID", how = "inner")

    clinvar_proteins.drop_duplicates(subset = ["VarID"], inplace=True)
    print(clinvar_proteins.head())


    clinvar_proteins["location"] = clinvar_proteins["location"].astype(int)
    clinvar_proteins["start"] = clinvar_proteins["protein_start_end"].str.split("_").str[1].astype(int)
    clinvar_proteins["end"] = clinvar_proteins["protein_start_end"].str.split("_").str[2].astype(int)

    subset_mapped_proteins = clinvar_proteins[
    (clinvar_proteins["location"] >= clinvar_proteins["start"]) &
    (clinvar_proteins["location"] <= clinvar_proteins["end"])
].drop_duplicates(subset = ["UniProtID", "start", "end", "location"])

    ### Now creating a feature table with MD data
    # Remove non-IDRs and HGMD labels
    h5py_path = os.path.abspath(config["h5py_path"])

    aa_order = "ARNDCQEGHILKMFPSTWYV"
    aaindex_res_mat = np.load(os.path.join(data_dir, "aa_index1.npy"))
    res_data, pair_data = load_MD_data(h5py_path)


    GPCRmd_labels = subset_mapped_proteins[(subset_mapped_proteins["MD Data Source"] == "GPCRmd") &
                                        (subset_mapped_proteins["changed_residue"] != "X")]

    unique_proteins = GPCRmd_labels["MD_protein_start_end"].unique()
    print("Unique protein regions", len(unique_proteins))
    unique_proteins_df = pd.DataFrame(unique_proteins, columns=["MD_protein_start_end"])



    train_proteins, val_proteins = train_test_split(unique_proteins_df, test_size=0.1, random_state=42)

    train_proteins_df = GPCRmd_labels[GPCRmd_labels["MD_protein_start_end"].isin(train_proteins["MD_protein_start_end"])]
    val_proteins_df = GPCRmd_labels[GPCRmd_labels["MD_protein_start_end"].isin(val_proteins["MD_protein_start_end"])]

    variant_information_columns = ["outcome", "UniProtID", "location", "MD_protein_start_end", "Original AA", 
                                "Changed AA"]
    location_column_name = "location"
    original_aa_column_name = None
    changed_aa_column_name = "changed_residue" 
    outcome_column_name = "outcome"
    train_feature_table = create_feature_table_GPCRmd(train_proteins_df, res_data, pair_data, aaindex_res_mat,
                                            aa_order, variant_information_columns,
                                            location_column_name,
                                            original_aa_column_name,
                                            changed_aa_column_name, 
                                            outcome_column_name,
                                            use_res_md=True, use_pair_md=True)
    val_feature_table = create_feature_table_GPCRmd(val_proteins_df, res_data, pair_data, aaindex_res_mat,
                                                aa_order, variant_information_columns,
                                                location_column_name,
                                                original_aa_column_name,
                                                changed_aa_column_name, 
                                                outcome_column_name, 
                                                use_res_md = True, use_pair_md = True)

    train_feature_table.to_csv(os.path.join(data_dir, "GPCRmd_train_val", "train.csv"))
    val_feature_table.to_csv(os.path.join(data_dir, "GPCRmd_train_val", "val.csv"))


if __name__ == "__main__":
    main()
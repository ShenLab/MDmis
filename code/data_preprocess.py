import torch 
import pandas as pd
import h5py
import numpy as np

###Date started: 05/23/24
###Purpose: To process Molecular Dynamics features for IDR, GPRC, and ATLAS data sources for input. And, to process ClinVar data ready for training.
pd.set_option('display.max_columns', 500)

## Defining paths
data_dir = "/share/vault/Users/az2798/"
protein_seq_mapping_dir = "/home/az2798/IDR_cons/data/"

def add_char_second_last(s):
    """ Adds a character to the second last location of a string """
    return s[:-1] + "_" + s[-1]

def read_protein_sequences():
    """ Reads the protein sequences and filters by source """
    protein_seqs = pd.read_csv(f'{protein_seq_mapping_dir}protein_sequences.csv', header=0)
    print(protein_seqs["source"].value_counts(), "All the protein seqs before splitting by source")
    return protein_seqs[protein_seqs["source"].isin(["IDRome", "GPCRmd", "ATLAS", "PED"])]

def read_atlas_metadata():
    """ Reads ATLAS metadata and modifies the PDB column """
    atlas_metadata = pd.read_csv(f'{protein_seq_mapping_dir}ATLAS.csv', header=0)
    atlas_metadata["PDB"] = atlas_metadata["PDB"].apply(add_char_second_last)
    return atlas_metadata

def split_sequences(protein_seqs):
    """ Splits sequences on '<linker>' and generates part names """
    linker_rows = protein_seqs['sequence'].str.contains('<linker>') | protein_seqs['modify_seq'].str.contains('<linker>')
    non_linker_rows = ~linker_rows

    linker_df = protein_seqs[linker_rows].copy()
    seq_split = linker_df['sequence'].str.split('<linker>')
    num_parts = seq_split.apply(len)
    other_cols_repeated = linker_df.loc[linker_df.index.repeat(num_parts)].reset_index(drop=True)
    part_numbers = num_parts.apply(lambda x: range(1, x + 1)).explode().reset_index(drop=True)
    other_cols_repeated['sequence'] = seq_split.explode().reset_index(drop=True)
    other_cols_repeated['split_name'] = other_cols_repeated['name'] + '_part' + part_numbers.astype(str)

    protein_seqs_split = pd.concat([other_cols_repeated, protein_seqs[non_linker_rows]]).sort_index().reset_index(drop=True)
    
    print(protein_seqs_split["source"].value_counts(), "All the protein seqs after splitting by source")

    return protein_seqs_split
def read_protein_mappings():
    """ Reads and processes protein mappings """
    protein_mappings = pd.read_csv(f'{protein_seq_mapping_dir}mmseqs_databases/mapping.m8', header=None,
                                   names=["query", "subject", "pident", "length", "mismatch", "gapopen", "qstart", 
                                          "qend", "sstart", "send", "evalue", "bitscore"], sep='\t')
    print(protein_mappings.shape, "Protein Mappings Shape before dropping duplicates")
    protein_mappings_deduplicated = protein_mappings.drop_duplicates(subset='query', keep='first')
    print(protein_mappings_deduplicated.shape, "Protein Mappings Shape after dropping duplicates")
    return protein_mappings_deduplicated
def merge_protein_sequences(protein_seqs_split, protein_mappings, atlas_metadata):
    """ Merges protein sequences with mappings and ATLAS metadata """
    split_rows = protein_seqs_split[ ( protein_seqs_split["split_name"].notna() ) &
                                    ( protein_seqs_split["source"].isin(["GPCRmd", "ATLAS", "PED"]) )
                                     ]
    non_split_rows = protein_seqs_split[ ( protein_seqs_split["split_name"].isna() ) &
                                       ( protein_seqs_split["source"].isin(["GPCRmd", "ATLAS", "PED"]) ) 
                                       ]

    merged_protein_seqs_nosplit = pd.merge(left=non_split_rows, right=protein_mappings[["query", "subject", "sstart", "send"]],
                                        how='inner', left_on="name", right_on="query")
    merged_protein_seqs_split = pd.merge(left=split_rows, right=protein_mappings[["query", "subject", "sstart", "send"]],
                                        how='inner', left_on="split_name", right_on="query")
   


    merged_protein_seqs = pd.concat([merged_protein_seqs_nosplit,
                                    merged_protein_seqs_split], ignore_index=True)
    merged_protein_seqs.rename({"subject":"UniProtID"}, axis = 1, inplace=True)


    merged_protein_seqs["protein_start_end"] = (merged_protein_seqs["UniProtID"] + "_" + 
                                                     merged_protein_seqs["sstart"].fillna(-1).astype(int).astype(str) + "_" + 
                                                     merged_protein_seqs["send"].fillna(-1).astype(int).astype(str))
    
    
    return merged_protein_seqs

def handle_idrome_proteins(protein_seqs_split):
    """ Handles IDRome proteins by adding a protein_start_end column """
    idr_protein_seqs = protein_seqs_split[protein_seqs_split["source"] == "IDRome"].copy()
    idr_protein_seqs["protein_start_end"] = idr_protein_seqs["name"]
    idr_protein_seqs["UniProtID"] = idr_protein_seqs["name"].str.split("_").str[0]
    return idr_protein_seqs

def combine_and_save_sequences(merged_protein_seqs, idr_protein_seqs):
    """ Combines the sequences and saves to CSV """
    final_protein_seqs = pd.concat([merged_protein_seqs, idr_protein_seqs], ignore_index=True)
    final_protein_seqs.to_csv(f'{protein_seq_mapping_dir}mapped_protein_seqs.csv')
    return final_protein_seqs

def print_distribution_by_source(final_protein_seqs):
    """ Prints the distribution of unique proteins by source """
    unique_proteins = final_protein_seqs.drop_duplicates(subset=["protein_start_end"])
    print("Number of proteins", unique_proteins["source"].shape)
    print("Distribution by source", unique_proteins["source"].value_counts())

# Main processing
protein_seqs = read_protein_sequences()
atlas_metadata = read_atlas_metadata()
protein_seqs_split = split_sequences(protein_seqs)
protein_seqs_split.to_csv(f'{protein_seq_mapping_dir}protein_sequences_split.csv', index=False)
protein_mappings = read_protein_mappings()
merged_protein_seqs = merge_protein_sequences(protein_seqs_split, protein_mappings, atlas_metadata)
idr_protein_seqs = handle_idrome_proteins(protein_seqs_split)
final_protein_seqs = combine_and_save_sequences(merged_protein_seqs, idr_protein_seqs)
print_distribution_by_source(final_protein_seqs)
#Mapping complete

## Part 2: Protein MD simulations data
accumulated_res_data = {}
accumulated_pair_data = {}

def identify_split_indices(sequence, marker='<linker>'):
    #Since the beginning and end of each peptide is marked by a -1, we can leverage the <linker> info to remove
    #the -1s and split the features per peptide
    protein_parts = sequence.split(marker)
    start_indices = []
    end_indices = []
    index_tracker = 1
    for part in protein_parts:
        start_indices.append(index_tracker)
        end_indices.append(index_tracker + len(part))
        index_tracker = index_tracker + len(part) + 2
    return start_indices, end_indices

def print_h5_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")    

def filter_split_h5_structure(name, obj, names_list, src):
    """ Filters and splits HDF5 datasets based on sequence names """
    if isinstance(obj, h5py.Dataset):
        for base_name in names_list:
            if name.startswith(base_name):
                sequence = protein_seqs[protein_seqs["name"] == base_name]["sequence"].values[0]
                data = src[name][:]
                start_indices, end_indices = identify_split_indices(sequence)
                
                for i, (start, end) in enumerate(zip(start_indices, end_indices)):
                    if "<linker>" in sequence:
                        match = final_protein_seqs[final_protein_seqs["split_name"] == f'{base_name}_part{i+1}']
                        if not match.empty:
                            protein_name_homology = match["protein_start_end"].values[0]
                        else:
                            break  # or some default value
                    else:
                        match = final_protein_seqs[final_protein_seqs["name"] == str(base_name)]
                        if not match.empty:
                            protein_name_homology = match["protein_start_end"].values[0]
                        else:
                            break  # or some default value
                    
                    if protein_name_homology:
                        print(protein_name_homology, "Protein Names")
                        if len(obj.shape) == 2:
                            segment_data = data[start:end, :]
                            new_name = f'{protein_name_homology}_res'
                            if new_name in accumulated_res_data:
                                accumulated_res_data[new_name].append(segment_data)
                            else:
                                accumulated_res_data[new_name] = [segment_data]
                        elif len(obj.shape) == 3:
                            segment_data = data[start:end, start:end, :]
                            new_name = f'{protein_name_homology}_pair'
                            if new_name in accumulated_pair_data:
                                accumulated_pair_data[new_name].append(segment_data)
                            else:
                                accumulated_pair_data[new_name] = [segment_data]
                break 
                    
def create_or_update_datasets(dst):
    for data_dicts in [accumulated_res_data, accumulated_pair_data]:
        for name, data_list in data_dicts.items():
            # Print the name and shapes of the data segments
            print(f"Processing dataset: {name}")
            shapes = [data.shape for data in data_list]
            print(f"Shapes of data segments: {shapes}")
            
            # Check if all elements in data_list have the same shape
            first_shape = data_list[0].shape
            if all(data.shape == first_shape for data in data_list):
                if len(data_list) > 1:
                    average_data = np.mean(data_list, axis=0)
                else:
                    average_data = data_list[0]

                if name in dst:
                    dst[name][:] = average_data
                else:
                    dst.create_dataset(name, data=average_data)
            else:
                print(f"Warning: Inconsistent shapes found in data segments for {name}. Skipping this dataset.")


names_list = protein_seqs["name"].tolist()

md_data_file = f"{data_dir}train_data_all/feature_all_ATLAS_GPCRmd_PED_IDRome_proteinflow_newres.h5"

filtered_md_data_file = f"{data_dir}train_data_all/filtered_feature_all_ATLAS_GPCRmd_IDRome.h5"

with h5py.File(md_data_file, "r") as src, h5py.File(filtered_md_data_file, "w") as dst:
    src.visititems(lambda name, obj: filter_split_h5_structure(name, obj, names_list, src ))
    create_or_update_datasets(dst)

## Part 3: Obtaining ClinVar and PrimateAI data for training
benign_path_data = pd.read_csv(f'{data_dir}ClinVar_Data/training.csv', header = 0, index_col = 0)
print(benign_path_data.shape, "Before removal of synonymous variants")
# Checking for synonymous variants
benign_path_data = benign_path_data[ benign_path_data["ref"] != benign_path_data["alt"] ]
print(benign_path_data.shape, "After removal of synonymous variants")



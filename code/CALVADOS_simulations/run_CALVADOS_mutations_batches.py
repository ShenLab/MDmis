import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
import scipy.stats as ss
import numpy as np
import os
import re
import warnings
import time
import argparse
from CALVADOS_code.CALVADOS_utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.rcParams.update({'font.size': 13})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import sys
import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.append(ROOT)
from utils import *
from config import config
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mut_type',
                         help = "Indicates which mutation type to run. Options include phr, plr, and ben.")
    parser.add_argument('-d', '--directory', 
                        help = "Indicates the target directory for the simulations.")
    parser.add_argument('-s', '--set_mutations', help= "Indicates the starting index for the batch of runs.")
    args = parser.parse_args()
    
    data_dir = os.path.abspath(config["data_dir"])
    results_dir = os.path.abspath(config["results_dir"])


    train_feature_table = pd.read_csv(
        os.path.join(data_dir, "clinical_train_val", "fold_1",
                        "train.csv"), 
        index_col = 0
    )
    val_feature_table = pd.read_csv(
        os.path.join(data_dir, "clinical_train_val", "fold_1",
                        "val.csv"), 
        index_col = 0
    )

    IDR_MD_features = pd.concat([train_feature_table, val_feature_table],
                                axis = 0)
    RMSF_column_name = "Res_MD_3_pos_0" # used earlier. You may use a Region Length column with a cutoff of 800

    IDR_MD_features['RMSF Category'] = np.select(
            [
                (IDR_MD_features["outcome"] == 1) & 
                (IDR_MD_features[RMSF_column_name] > 4.5),
                (IDR_MD_features["outcome"] == 1) &
                (IDR_MD_features[RMSF_column_name] <= 4.5),
                IDR_MD_features["outcome"] == 0
            ],
            ['Pathogenic - High RMSF', 'Pathogenic - Low RMSF', 'Benign']
        )


    IDR_MD_features = IDR_MD_features.sample(frac=1,
                                            random_state = 42).reset_index(drop=True) #randomly shuffle the data


    #starting with top 50
    MD_metadata = pd.read_csv("/home/az2798/MDmis/data/MD_metadata.csv",
                            index_col= 0 )

    IDRome_metadata = MD_metadata[MD_metadata["source"]== "IDRome"]

    #Starting with Pathogenic - High RMSF
    #Pathogenic_high_rmsf = IDR_MD_features[IDR_MD_features["RMSF Category"] == "Pathogenic - High RMSF"].head(50)
    if args.mut_type == "phr":
        variants_to_simulate = IDR_MD_features[IDR_MD_features["RMSF Category"] == "Pathogenic - High RMSF"]
    
    elif args.mut_type == "plr":
        variants_to_simulate = IDR_MD_features[IDR_MD_features["RMSF Category"] == "Pathogenic - Low RMSF"]
    
    else:
        variants_to_simulate = IDR_MD_features[IDR_MD_features["RMSF Category"] == "Benign"].head(800)

    # Step 1: Find the sequence in the IDRome_metadata
    processed_sequences = []

    for index, row in variants_to_simulate.iterrows():

        uniprot_id = row["UniProtID"]    
        location = row["location"]  
        
        matching_metadata = IDRome_metadata[IDRome_metadata["UniProtID"] == uniprot_id]
        for meta_index, meta_row in matching_metadata.iterrows():
            protein_start_end = meta_row["protein_start_end"]
            _, start, end = protein_start_end.split("_")
            start, end = int(start), int(end)
            if location > end or location < start:
                #print(f"Skipping {protein_start_end} since variant {location} falls outside.")
                continue
            else:
                
                location_index = location - start
                sequence = meta_row["sequence"]
                modified_subsequence = (
                    sequence[:location_index] + 
                    row["Changed AA"] + 
                    sequence[location_index + 1:]
                )
                processed_sequences.append(
                    {
                    "UniProtID": uniprot_id,
                    "original_sequence": sequence,
                    "modified_sequence": modified_subsequence,
                    "location": location,
                    "protein_start_end": protein_start_end,
                    "modified_ID": f"{protein_start_end}_{row['Original AA']}:{location}:{row['Changed AA']}",
                    "RMSF Category": variants_to_simulate.head(1)["RMSF Category"].values[0]
                    }
                )
    processed_sequences_df = pd.DataFrame(processed_sequences)
    processed_sequences_df.to_csv(
        f"{data_dir}calvados_mutations_{args.mut_type}_{args.set_mutations}.csv", index=False)

    #results_dir = "/share/vault/Users/az2798/CALVADOS_runs/Pathogenic_High_RMSF_Shannon/"
    results_dir = args.directory
    residues_file = "/home/az2798/MDmis/residues.csv"
    # Step 2: run CALVADOS for processed sequences
    for modified_sequences in processed_sequences[int(args.set_mutations)-1:min(int(args.set_mutations)+19, len(processed_sequences))]:
        time_start = time.time()
        print(f"Running {modified_sequences['modified_ID']}.")
        run_md_sim(modified_sequences["modified_ID"], 
                modified_sequences["modified_sequence"],
                residues_file,
                results_dir,
                charged_N_terminal_amine=True,
                charged_C_terminal_carboxyl = False,
                charged_histidine = False)
        time_end = time.time()
        
        print(f"Finished {modified_sequences['modified_ID']} in {time_end - time_start} seconds.")


if __name__ == "__main__":
    main()

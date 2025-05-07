import pandas as pd
import numpy as np
import glob
pd.set_option('display.max_columns', 500)
import os
import sys
import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.append(ROOT)
from config import config

def filter_merge_dbNSFP( list_of_dbNSFP_files, protein_info_columns, metrics_columns,
                        aggregate):

    #Iterate through each dbNSFP file and retain the proteins. 
    list_of_dfs = []
    for file in list_of_dbNSFP_files:
        temporary_chromosome_df = pd.read_csv(file, sep = "\t", header=0)
        temporary_chromosome_df = temporary_chromosome_df[protein_info_columns + metrics_columns]
        #For each protein and reference position, compute the conservation
        
        for col in metrics_columns:
            temporary_chromosome_df[col] = pd.to_numeric(temporary_chromosome_df[col], errors='coerce')

        temporary_chromosome_df = temporary_chromosome_df.dropna(subset=metrics_columns)
        print(temporary_chromosome_df.head(10))
        if aggregate:
            aggregated_df = temporary_chromosome_df.groupby(protein_info_columns, as_index=False)[metrics_columns].mean()
            print(aggregated_df.head(10))

            list_of_dfs.append(aggregated_df)
        else:
            print(temporary_chromosome_df.head(10))

            list_of_dfs.append(temporary_chromosome_df)
    merged_df = pd.concat(list_of_dfs, ignore_index=True)
    return merged_df
def main():
    vault_dir = os.path.abspath(config["vault_dir"])
    ##### Protein Location Constraint data
    list_of_dbNSFP_files = glob.glob(os.path.join(vault_dir,"dbNSFP4.8a_variant*") ) 
    protein_info_columns = ["aaref", "aapos", "Ensembl_transcriptid", "Uniprot_acc"]
    constraint_metrics_columns = ["GERP++_NR", "GERP++_RS", "GERP++_RS_rankscore",
                             "phyloP470way_mammalian"]
    merged_df = filter_merge_dbNSFP( list_of_dbNSFP_files, protein_info_columns,
                                    constraint_metrics_columns, aggregate=True)
    merged_df.to_csv(os.path.join(vault_dir,"dbNSFP4_merged.csv"))

if __name__ == "__main__":
    main()

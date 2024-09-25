import pandas as pd
import numpy as np
import glob
pd.set_option('display.max_columns', 500)

def filter_merge_dbNSFP( list_of_dbNSFP_files, columns_to_keep, columns_to_aggregate):
    #First, obtain unique proteins from the clinvar_df

    #unique_proteins_in_clinvar = clinvar_df["uniprotID"].unique()

    #Second, iterate through each dbNSFP file and retain those proteins. 
    list_of_dfs = []
    for file in list_of_dbNSFP_files:
        temporary_chromosome_df = pd.read_csv(file, sep = "\t", header=0)
        temporary_chromosome_df = temporary_chromosome_df[columns_to_keep + columns_to_aggregate]
        #For each protein and reference position, compute the conservation
        
        for col in columns_to_aggregate:
            temporary_chromosome_df[col] = pd.to_numeric(temporary_chromosome_df[col], errors='coerce')

        temporary_chromosome_df = temporary_chromosome_df.dropna(subset=columns_to_aggregate)

        aggregated_df = temporary_chromosome_df.groupby(columns_to_keep, as_index=False)[columns_to_aggregate].mean()

        print(aggregated_df.head(30))

        list_of_dfs.append(aggregated_df)
    merged_df = pd.concat(list_of_dfs, ignore_index=True)
    return merged_df
def main():
    #clinvar_df = pd.read_csv("/share/vault/Users/az2798/ClinVar_Data/training.csv")
    list_of_dbNSFP_files = glob.glob("/share/vault/Users/az2798/GERP_data/dbNSFP4.8a_variant*")  
    columns_to_keep = ["aaref", "aapos", "Ensembl_transcriptid", "Uniprot_acc"]
    columns_to_aggregate = ["GERP++_NR", "GERP++_RS", "GERP++_RS_rankscore"]
    merged_df = filter_merge_dbNSFP( list_of_dbNSFP_files, columns_to_keep,
                        columns_to_aggregate)
    merged_df.to_csv("/share/vault/Users/az2798/GERP_data/dbNSFP4_merged.csv")

if __name__ == "__main__":
    main()

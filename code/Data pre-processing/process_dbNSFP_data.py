import pandas as pd
import numpy as np
import glob
pd.set_option('display.max_columns', 500)

def filter_merge_dbNSFP( list_of_dbNSFP_files, protein_info_columns, metrics_columns,
                        aggregate):
    #First, obtain unique proteins from the clinvar_df

    #unique_proteins_in_clinvar = clinvar_df["uniprotID"].unique()

    #Second, iterate through each dbNSFP file and retain those proteins. 
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
    #clinvar_df = pd.read_csv("/share/vault/Users/az2798/ClinVar_Data/training.csv")
    
    ##### Protein Location Constraint data
    list_of_dbNSFP_files = glob.glob("/share/vault/Users/az2798/dbNSFP_data/dbNSFP4.8a_variant*")  
    protein_info_columns = ["aaref", "aapos", "Ensembl_transcriptid", "Uniprot_acc"]
    constraint_metrics_columns = ["GERP++_NR", "GERP++_RS", "GERP++_RS_rankscore",
                             "phyloP470way_mammalian"]
    merged_df = filter_merge_dbNSFP( list_of_dbNSFP_files, protein_info_columns,
                                    constraint_metrics_columns, aggregate=True)
    merged_df.to_csv("/share/vault/Users/az2798/dbNSFP_data/dbNSFP4_merged.csv")


    ##### Protein Location AND Variant Pathogenicity measures
    protein_info_columns = ["aaref", "aaalt", "aapos", "Ensembl_transcriptid", "Uniprot_acc"]

    pathogenicity_measures = ["gMVP_score", "AlphaMissense_score", 
                              "EVE_score", "ESM1b_score"]
    variant_predictions_df = filter_merge_dbNSFP(list_of_dbNSFP_files, protein_info_columns,
                        pathogenicity_measures, aggregate=False)
    variant_predictions_df.to_csv("/share/vault/Users/az2798/dbNSFP_data/variant_predictions.csv")

if __name__ == "__main__":
    main()

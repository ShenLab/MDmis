import re
import os
import numpy as np
import pandas as pd
import scipy.stats as ss

def main():
    ## Preliminary data loading and processing
    data_dir = "/home/az2798/MDmis/data/"
    results_dir = "/home/az2798/MDmis/results/clinical_figures"
    vault_dir = "/share/vault/Users/az2798/"

    RMSF_column_name = "Res_MD_3"
    IDRs_table = pd.read_csv(
        os.path.join(
            data_dir, "clinical_train_val", "feature_table.csv"
        ) 
        , index_col= 0, low_memory= False
    )
   
    
    IDRs_table["start"] = IDRs_table["protein_start_end"].str.split("_").str[1].astype(int)
    IDRs_table["end"] = IDRs_table["protein_start_end"].str.split("_").str[2].astype(int)

    IDRs_table["Region Length"] = IDRs_table["end"] - IDRs_table["start"] + 1

    IDRs_table["Variant Effect"] = np.where(IDRs_table["outcome"] == 1, "Pathogenic", "Benign")

    IDRs_table = IDRs_table[IDRs_table[RMSF_column_name] <20]
    IDRs_table['Length Category'] = np.select(
        [
            (IDRs_table["Variant Effect"] == "Pathogenic") & 
            (IDRs_table["Region Length"] > 800),
            (IDRs_table["Variant Effect"] == "Pathogenic") &
            (IDRs_table["Region Length"] <= 800),
            IDRs_table["Variant Effect"] == "Benign"
        ],
        ['Pathogenic - Long IDRs', 'Pathogenic - Short IDRs', 'Benign']
    )

    ############## Introducing the signal, transit, and propeptide tracks
    def get_domain_type(row):
        """ Function to find domain type based on UniProtID and location """
        matches = up_tracks[(up_tracks["UniProtID"] == row["UniProtID"]) & 
                            (up_tracks["Start"] <= row["location"]) & 
                            (up_tracks["End"] >= row["location"])]
        if not matches.empty:
            return matches["Domain_Type"].iloc[0], matches["End"].iloc[0]
        else:
            return "Chain/Other", None
         
    up_tracks = pd.read_csv(os.path.join(data_dir, "UniProt_tracks_processed.csv"), index_col =0)
    
    IDRs_table[["Molecule Processing", "Processing End Site"]] = IDRs_table.apply(get_domain_type, axis=1,result_type="expand")

    print(IDRs_table.shape)
    contingency_table = pd.crosstab(IDRs_table['Molecule Processing'], IDRs_table['Length Category'])
    print(contingency_table)
    print(ss.chi2_contingency(contingency_table))


    IDRs_table["Distance_from_end"] = IDRs_table["Processing End Site"] - IDRs_table["location"]
    
    print(np.mean(IDRs_table[(IDRs_table["Molecule Processing"] =="signal") | 
                             (IDRs_table["Molecule Processing"] =="transit")]["Distance_from_end"]))
    
if __name__ == "__main__":
    main()
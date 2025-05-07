import re
import os
import numpy as np
import pandas as pd
import scipy.stats as ss
import sys
import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.append(ROOT)
from config import config

def main():
    pd.set_option("display.max_columns", 100)
    ## Preliminary data loading and processing
    data_dir = os.path.abspath(config["data_dir"])
    vault_dir = os.path.abspath(config["vault_dir"])

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
    
    all_labels = pd.read_csv(
        os.path.join(vault_dir, "ClinVar_Data","training.csv"),
          index_col=0, low_memory=False
    )

    all_labels.rename(columns={"pos.orig":"location", "ref": "Original AA",
                               "alt": "Changed AA", "uniprotID": "UniProtID",
                               "CLNREVSTAT": "ClinVar Review Status"},
                               inplace=True)
    all_labels = all_labels[all_labels["ClinVar Review Status"] != "."]
    all_labels = all_labels[["location", "Original AA",
                             "Changed AA", "UniProtID", 
                             "ClinVar Review Status"]]
    IDR_clinvar_labels_only = pd.merge(IDRs_table, all_labels,
             left_on=["UniProtID", "location", "Changed AA"],
             right_on= ["UniProtID", "location", "Changed AA"],
             how="inner")

    
    clinv_review_contingency_table = pd.crosstab(IDR_clinvar_labels_only["Length Category"],
                IDR_clinvar_labels_only["ClinVar Review Status"])
    print(clinv_review_contingency_table)
    print(
        ss.chi2_contingency(clinv_review_contingency_table))

if __name__ == "__main__":
    main()
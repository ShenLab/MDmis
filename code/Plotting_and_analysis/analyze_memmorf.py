import re
import os
import numpy as np
import pandas as pd
import scipy.stats as ss
import glob
import json
import sys
import subprocess
import pathlib
ROOT = pathlib.Path(__file__).parents[1]
sys.path.append(str(ROOT))
from config import config


def calculate_percent_overlap(row):
    """
    Calculates the percentage of overlap between the two domains.

    Args:
        row: DataFrame row

    Returns:
        The percentage of overlap between the two intervals, or 0 if there is no overlap.
    """
    start1 = row["start"]
    start2 = row["Domain_Start"]
    end1 = row["end"]
    end2 = row["Domain_End"]
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = max(0, overlap_end - overlap_start)

    length1 = end1 - start1
    length2 = end2 - start2

    if overlap_length > 0:
        percent_overlap = (overlap_length / min(length1, length2)) * 100
    else:
        percent_overlap = 0

    return percent_overlap


def main():


    ## Preliminary data loading and processing
    data_dir = os.path.abspath(config["data_dir"])
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
        ['Pathogenic >800aa IDRs', 'Pathogenic <800aa IDRs', 'Benign']
    )


    ############## Identifying overlapping domains
    memmorf_df = pd.read_csv(os.path.join(config["data_dir"], "memmorf_db.tsv"),
                             sep = "\t", header = None
                             )
    column_names = ["UniProt_ID", "PDB_ID", "x", "set", "-", "chains",
                                      "Start", "End", "type", "localization"]
    memmorf_df = memmorf_df.rename({memmorf_df.columns[i]: column_names[i] for i in range(10)},
                                   axis=1)
    
    memmorf_df = memmorf_df[memmorf_df["Start"]!="-"]

    memmorf_df["Start"] = memmorf_df["Start"].astype(int)
    memmorf_df["End"] = memmorf_df["End"].astype(int)
    print(memmorf_df.head())

    def get_memmorf_domain(row):
        """ Function to find MemMorF domain based on UniProtID and location """
        matches = memmorf_df[(memmorf_df["UniProt_ID"] == row["UniProtID"]) & 
                            (memmorf_df["Start"] <= row["location"]) & 
                            (memmorf_df["End"] >= row["location"])]
        if not matches.empty:
            return matches["type"].iloc[0], matches["Start"].iloc[0], matches["End"].iloc[0]
        else:
            return "No Domain", None, None   
        
    IDRs_table[["MemMorF_Domain", "Domain_Start", "Domain_End"]] = IDRs_table.apply(get_memmorf_domain, axis=1,result_type="expand")
    print(IDRs_table.head())
    contingency_table = pd.crosstab(IDRs_table["MemMorF_Domain"], IDRs_table['Length Category'])
    print(contingency_table)
    print(ss.chi2_contingency(contingency_table))


    IDRs_table["Overlap percent"] = IDRs_table.apply(calculate_percent_overlap, axis=1, result_type = "expand")
    
    print(IDRs_table.groupby("Length Category")["Overlap percent"].mean())

    


if __name__ == "__main__":
    main()
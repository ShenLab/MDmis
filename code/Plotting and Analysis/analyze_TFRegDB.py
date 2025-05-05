import re
import os
import numpy as np
import pandas as pd
import scipy.stats as ss



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

    ############## Identifying overlapping domains

    def get_TF_domain(row):
        """ Function to find TF domain based on UniProtID and location """
        matches = TF_domains[(TF_domains["Uniprot ID"] == row["UniProtID"]) & 
                            (TF_domains["Start"] <= row["location"]) & 
                            (TF_domains["End"] >= row["location"])]
        if not matches.empty:
            return matches["Domain type"].iloc[0], matches["Start"].iloc[0], matches["End"].iloc[0]
        else:
            return "No Domain", None, None   
        
    TF_domains = pd.read_csv(os.path.join(data_dir, "TFRegDB_table.csv"), index_col =0)
    TF_domains["Start"] = TF_domains["Coordinates"].str.split("-").str[0].astype(int)
    TF_domains["End"] = TF_domains["Coordinates"].str.split("-").str[1].astype(int)
    print(TF_domains.head(50))

    IDRs_table[["TF_Domain", "Domain_Start", "Domain_End"]] = IDRs_table.apply(get_TF_domain, axis=1,result_type="expand")
    print(IDRs_table.shape)

    
    contingency_table = pd.crosstab(IDRs_table["TF_Domain"], IDRs_table['Length Category'])
    print(contingency_table)
    print(ss.chi2_contingency(contingency_table))



    IDRs_table["Overlap percent"] = IDRs_table.apply(calculate_percent_overlap, axis=1, result_type = "expand")
    
    print(IDRs_table.groupby("Length Category")["Overlap percent"].mean())
    
    IDRs_table[(IDRs_table["TF_Domain"] != "No Domain") &
                            (IDRs_table["Length Category"] == "Pathogenic - Short IDRs")][["UniProtID", 
                                                                                          "location","protein_start_end", "TF_Domain", "Original AA",
                                                                                          "Changed AA"]].to_csv(os.path.join(data_dir, "TFRegDB_Short_IDRs.csv"))
if __name__ == "__main__":
    main()
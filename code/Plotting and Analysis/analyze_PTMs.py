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
    
    curated = True
    ###### PTM database merging
    # CURATED DATABASE
    if curated:
        PTM_dataframes = []
        ptm_column_names = ["UniProt_ID", "Location", "PTM_Type"]

        for filename in os.listdir(os.path.join(vault_dir, "PTM_data")):
            file_path = os.path.join(vault_dir, "PTM_data", filename)
            extracted_rows = []
            with open(file_path, 'r') as f:
                lines = f.readlines()[2:]  #Skip the header line
            for line in lines:
                parts = line.strip().split()
                if "_" in parts[0]:
                    #This implies that the first column is Gene_Name
                    extracted_rows.append(parts[1:4])
                elif re.match("^[a-zA-Z]+.*", parts[0]):
                    #This implies that the first column is UniProtID
                    extracted_rows.append(parts[0:3])
                else:
                    continue
            df = pd.DataFrame(extracted_rows, columns=ptm_column_names)
            df["UniProt_ID"] = df["UniProt_ID"].str.split('-').str[0]
            PTM_dataframes.append(df)

        #PTM_combined_df = pd.DataFrame(PTM_dataframes, columns = ptm_column_names)
        PTM_combined_df = pd.concat(PTM_dataframes, ignore_index=True)
        print(PTM_combined_df.head())

        PTM_combined_df["Location"] = PTM_combined_df["Location"].astype(int)
        PTM_combined_df.drop_duplicates(inplace=True)
        PTM_merged = pd.merge(left = IDRs_table,
                          right = PTM_combined_df,
                          left_on=["UniProtID", "location"],
                          right_on=["UniProt_ID", "Location"],
                          how="left",
                          suffixes= ["", "_y"])



    #USE PREDICTIONS
    else:
        PTM_pred_df = pd.read_csv("/home/ch3849/tool_code/MIND/temp/correct_predictions.csv")
        PTM_pred_df = PTM_pred_df[PTM_pred_df["pred_score"]> 0.5]
        PTM_merged = pd.merge(left = IDRs_table,
                          right = PTM_pred_df,
                          left_on=["UniProtID", "location"],
                          right_on=["uid", "site"],
                          how="left",
                          suffixes= ["", "_y"])
        PTM_merged.rename(columns={"PTM_type": "PTM_Type"}, inplace=True)
    #print(PTM_combined_df[["UniProt_ID", "Location"]].value_counts())
    
    #PTM_combined_df.drop_duplicates(subset=["UniProt_ID", "Location"], inplace=True)

    
    
    print(PTM_merged.shape)


    PTM_merged["PTM_Type"] = PTM_merged["PTM_Type"].fillna("None")
    PTM_merged["PTM"] = np.where(PTM_merged["PTM_Type"] =="None", "No", "Yes")
    

    PTM_merged[(PTM_merged["PTM_Type"] == "Phosphorylation") &
                            (PTM_merged["Length Category"] == "Pathogenic - Short IDRs")][["UniProtID", 
                                                                                          "location","protein_start_end", "PTM_Type", "Original AA",
                                                                                          "Changed AA"]].to_csv(os.path.join(data_dir, "phosphorylation_sites_IDRs.csv"))
    
    #.drop_duplicates(subset = ["UniProtID", "start", "end", "location"])

    


    
    contingency_table = pd.crosstab(PTM_merged['PTM'], PTM_merged['Length Category'])
    print(contingency_table)
    print(ss.chi2_contingency(contingency_table))

    print(PTM_merged[(PTM_merged["PTM"] == "Yes") & 
                            (PTM_merged["Length Category"] == "Pathogenic - Short IDRs")]["PTM_Type"].value_counts(ascending=False))
    print(PTM_merged[(PTM_merged["PTM_Type"] == "Phosphorylation") & 
                            (PTM_merged["Length Category"] == "Pathogenic - Short IDRs")]["Changed AA"].value_counts(ascending=False))
    
    sty_residues = {"S", "T", "Y"}
    def classify_mutation_type(row):
        original_in_sty = row["Original AA"] in sty_residues
        changed_in_sty = row["Original AA"] in sty_residues
        
        if original_in_sty and changed_in_sty:
            return "STY --> STY"
        elif original_in_sty and not changed_in_sty:
            return "STY --> Non-STY"
        elif not original_in_sty and changed_in_sty:
            return "Non-STY --> STY"
        else:
            return "Non-STY --> Non-STY"

    PTM_merged['mutation_type_sty'] = PTM_merged.apply(classify_mutation_type, axis=1)
    contingency_table = pd.crosstab(PTM_merged[PTM_merged["PTM_Type"] == "Phosphorylation"]['mutation_type_sty'],
                                  PTM_merged["Length Category"])
    print(contingency_table)
    print(ss.chi2_contingency(contingency_table))

if __name__ == "__main__":
    main()
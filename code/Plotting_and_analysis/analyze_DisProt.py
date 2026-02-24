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
from CALVADOS_simulations.process_md_trajectory import *


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


    ### Subset UniProt sequences to generate a fasta for MSA generation
    print(IDRs_table[IDRs_table["Length Category"]=="Pathogenic >800aa IDRs"]["UniProtID"].unique())

    ############## Identifying overlapping domains
    disprot_df_array = []
    with open(os.path.join(data_dir, "DisProt", "disprot_IDpredictions.mjson"), "r") as f:
        for line in f:
            temp_dict = json.loads(line)
            temp_df = pd.DataFrame(temp_dict["predictions"][0]["scores"])
            temp_df["Uniprot_ID"] = temp_dict["id"]
            temp_df["Position"] = temp_df.index + 1 
            disprot_df_array.append(temp_df)
    disprot_df = pd.concat(disprot_df_array, axis=0, ignore_index=True)

    disprot_df.columns = [ "DisProt_Score", "Uniprot_ID", "Position"]
    disprot_df["Position"] = disprot_df["Position"].astype(int)

    disprot_merged = pd.merge(left = IDRs_table,
                        right = disprot_df,
                        left_on=["UniProtID", "location"],
                        right_on=["Uniprot_ID", "Position"],
                        how="left",
                        suffixes= ["", "_y"])
    
    print(disprot_merged[disprot_merged["DisProt_Score"].notna()].groupby("Length Category")["DisProt_Score"].describe())
    print(disprot_merged[disprot_merged["DisProt_Score"].notna()].groupby("Length Category")["Uniprot_ID"].value_counts()["Pathogenic <800aa IDRs"].sort_values(ascending=False).head(5))


if __name__ == "__main__":
    main()
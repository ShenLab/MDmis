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
import tqdm
ROOT = pathlib.Path(__file__).parents[1]
sys.path.append(str(ROOT))
from config import config
from utils import *


def main():
    ## Preliminary data loading and processing
    data_dir = os.path.abspath(config["data_dir"])
    results_dir = os.path.abspath(config["results_dir"])
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
            (IDRs_table["Variant Effect"] == "Benign") & 
            (IDRs_table["Region Length"] > 800),
            (IDRs_table["Variant Effect"] == "Benign") &
            (IDRs_table["Region Length"] <= 800)
        ],
        ['Pathogenic >800aa', 'Pathogenic <=800aa', 'Benign >800aa',
        "Benign <=800aa"]
    )
    length_palette = {"Pathogenic >800aa": "#e81a1a", "Pathogenic <=800aa": "#f5ed11",
                      "Benign >800aa": "#70bafa",  'Benign <=800aa': "#188ff5"}
    ##
    morf_chibi_files = glob.glob(os.path.join(config["morfchibi_dir"], "*"))

    morf_dfs = []
    for file in tqdm.tqdm(morf_chibi_files, desc = "Parsing MoRF Files"):
        if os.path.basename(file) != "timing.csv":
            uniprot_start_end = os.path.basename(file).split(" ")[0]
            df = pd.read_csv(file, sep = "\t", names=["Index", "Ref", "Probability"],
            dtype={"Index": int, "Probability": float}, skiprows=1)
            df["Uniprot_start_end"] = uniprot_start_end
            df["UniProtID"] = uniprot_start_end.split("_")[0]
            df["Start"] = int(uniprot_start_end.split("_")[1])
            df["End"] = int(uniprot_start_end.split("_")[2])
            df["Index"] = df["Index"]-1
            df["Location"] = df["Start"] + df["Index"]
            morf_dfs.append(df)

    morf_df_concat = pd.concat(morf_dfs, ignore_index=True)
    print(morf_df_concat.head())

    IDR_morf_merged = pd.merge(IDRs_table, morf_df_concat, how= "inner", left_on=["UniProtID", "location"],
    right_on=["UniProtID", "Location"], suffixes=["", "_y"])

    print(IDR_morf_merged.groupby("Length Category")["Probability"].describe())

    plot_ridgeplot(
        data=IDR_morf_merged,
        group_col='Length Category',
        value_col="Probability",
        results_dir=results_dir,
        xlabel = "MoRF Probability",
        plot_filename="MoRF_Prob.png", palette = length_palette,label_placement = "right",
        xlim =(-0.1, 1),
        bar_height = 0.05, xlim_buffer = 1.025
    )
    plt.clf()
if __name__ == "__main__":
    main()
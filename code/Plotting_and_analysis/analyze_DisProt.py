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


    ### Find disorder to transition qualitatively 
    P02461_pdb = "/nfs/user/Users/ch3849/descartes/prodance/disorder/IDRome/cg2all/61/P02461_79_1230.pdb"
    P02461_dcd = "/nfs/user/Users/ch3849/descartes/prodance/disorder/IDRome/cg2all/61/P02461_79_1230.dcd"
    traj = mdtraj.load_dcd(P02461_dcd, 
                            P02461_pdb)
    total_frames = traj.n_frames
    #frame_threshold_mdtraj = int(total_frames * 0.02)
    #sampled_frames = random.sample(range(frame_threshold_mdtraj, total_frames), k = num_sample_frames)
    traj  # sample frames by burning first threshold number and picking from the remainder

    dssp = mdtraj.compute_dssp(traj[10:], simplified=False)
    dssp = pd.DataFrame(dssp)

    dssp_curr = dssp.iloc[1:].reset_index(drop=True)
    dssp_prev = dssp.iloc[:-1].reset_index(drop=True)
    dssp_after = dssp.iloc[2:].reset_index(drop=True)
    is_T_after = dssp_after == "T"
    is_G_now = dssp_curr == "G"
    was_T_before = dssp_prev == "T"

    # Case 1: G now, T before
    case1 = (is_G_now & is_T_after).to_numpy().sum()
    case2 = (is_G_now & ~is_T_after).to_numpy().sum()
    total_G = is_G_now.to_numpy().sum()
    prop_case1 = case1 / total_G if total_G > 0 else np.nan
    prop_case2 = case2 / total_G if total_G > 0 else np.nan

    print(f"Case 1 (G preceded by T): {case1} ({prop_case1:.3f})")
    print(f"Case 2 (G preceded by not-T): {case2} ({prop_case2:.3f})")

    print(dssp[dssp.iloc[:, 597-1]=="G"])
    print(dssp[dssp.iloc[:, 438-1]=="G"])
    print(dssp[dssp.iloc[:, 606-1]=="G"])


    
    dssp_count = pd.DataFrame(
        [dssp[i].value_counts() for i in dssp.columns],
        columns=["B", "E", "H", "I", "G", "S", "T", " "],
        index=dssp.columns,
    )
    dssp_count.columns = ["dssp_" + i for i in dssp_count.columns]
    # ##### CG to All atom confirmation
    # quantile = 0.02
    # num_sample_frames = 800
    # getcontacts_script = os.path.join(config["code_dir"], "CALVADOS_simulations", "run_GetContacts.sh")
    # IDR_CG_dir = os.path.join(config["vault_dir"], "IDR_CG")
    # for protein in glob.glob(f'{IDR_CG_dir}/*'):

    #     traj = mdtraj.load_xtc(os.path.join(protein, "traj.xtc"), 
    #                            os.path.join(protein, "top_ca.pdb"))
    #     total_frames = traj.n_frames
    #     frame_threshold_mdtraj = int(total_frames * quantile)
    #     sampled_frames = random.sample(range(frame_threshold_mdtraj, total_frames), k = num_sample_frames)
    #     final_traj = traj[sampled_frames]  # sample frames by burning first threshold number and picking from the remainder

    #     dssp = mdtraj.compute_dssp(final_traj, simplified=False)
    #     dssp = pd.DataFrame(dssp)
    #     print(dssp)
    #     dssp_count = pd.DataFrame(
    #         [dssp[i].value_counts() for i in dssp.columns],
    #         columns=["B", "E", "H", "I", "G", "S", "T", " "],
    #         index=dssp.columns,
    #     )
    #     dssp_count.columns = ["dssp_" + i for i in dssp_count.columns]



if __name__ == "__main__":
    main()
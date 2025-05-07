import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from utils import *
import tqdm
from predict_MDmis import *
import pickle
import multiprocessing as mp
import argparse
import scipy.stats as ss
import pathlib
import sys
ROOT = pathlib.Path(__file__).parent
sys.path.append(ROOT)
from config import config

from feature_extraction_MSA import *

def generate_predictions(IDRome_metadata, res_data, pair_data,
                          aa_res_matrix, 
                        aa_order, MDmis_model,
                        result_dir = None):
    
    if result_dir is None:
        ValueError("Please enter a valid value for the result_dir")
    
    aa_to_index = {aa: idx for idx, aa in enumerate(aa_order)}

    for index, row in tqdm.tqdm(IDRome_metadata.iterrows()):
        IDR_protein_start_end = row["protein_start_end"]

        if os.path.exists(os.path.join(result_dir, IDR_protein_start_end, "features_predictions.csv")):
            print(f"Skipping {IDR_protein_start_end}, predictions already exist.")
            continue 
        os.makedirs(os.path.join(result_dir, IDR_protein_start_end), exist_ok=True)

        (UniProt_ID, protein_start, protein_end) = IDR_protein_start_end.split("_")
        print("protein_length",int(protein_end) - int(protein_start))
        print(len(row["sequence"]))

        rows_of_data = []
        for i in range(int(protein_end) - int(protein_start) + 1):
            #protein_start + i is the location
            location_variant = int(protein_start) + i


            if IDR_protein_start_end not in res_data or IDR_protein_start_end not in pair_data:
                continue
            
            residue_array = res_data[IDR_protein_start_end].copy()

            average_RMSF = np.mean(residue_array[:, 2])
            residue_array[:, 2] /= average_RMSF  #Normalize RMSF



            residue_md_features, residue_md_labels = get_window_with_padding(
                    residue_array,
                    i, 3
                    )

            interaction_cutoff = 0.4
            pair_data_above_cutoff = pair_data[IDR_protein_start_end] > interaction_cutoff
            count_pair_md_features = (
            np.sum(pair_data_above_cutoff[i, :, :9], axis=(0)) + 
            np.sum(pair_data_above_cutoff[:, i, :9], axis=(0))   
            )
            z_scored_cov = ss.zscore(pair_data[row["protein_start_end"]][: , :, -1], axis=None)
            avg_cov = (np.mean(z_scored_cov[i,:]) + np.mean(z_scored_cov[:, i])) /2
            original_aa = row["sequence"][i]
            
            for changed_aa in aa_order:
                if (changed_aa == original_aa):
                    continue
                residue_aa_index_features = aa_res_matrix[aa_to_index[changed_aa], :] - aa_res_matrix[aa_to_index[original_aa]]

                rows_of_data.append({
                    **{f"Res_AA_{j+1}": val for j, val in enumerate(residue_aa_index_features)},
                    **{label: val for label, val in zip(residue_md_labels, residue_md_features)},
                    **{f"Pair_MD_{j+1}": val for j, val in enumerate(count_pair_md_features)},
                    "Pair_MD_10": avg_cov,
                    "UniProtID": UniProt_ID,
                    "Location": location_variant,
                    "Original AA": original_aa,
                    "Changed AA": changed_aa
                })
        if rows_of_data:
            predictions_df = pd.DataFrame(rows_of_data)
            if isinstance(MDmis_model, list) & (len(MDmis_model)>1):
                #ensemble model
                predictions_dict = {}

                for k, model in enumerate(MDmis_model):
                    predictions_dict[f"MDmis_scores_{k}"] = predict_MDmis(
                        model, predictions_df, use_ESM_embed=False, use_conf_prop=False, use_Cons=False, use_ESM1b=False
                    )

                predictions_df = pd.concat([predictions_df, pd.DataFrame(predictions_dict)], axis=1)
            else:
                predictions_df["MDmis_scores"] = predict_MDmis(
                        model,  predictions_df, use_ESM_embed= False, use_conf_prop = False,use_Cons = False,
                        use_ESM1b = False
                    )
            MDmis_columns = [col for col in predictions_df.columns if "MDmis_scores" in str(col)]
            MDmis_columns = MDmis_columns + ["UniProtID", "Location", "Original AA", "Changed AA"]
            predictions_df_scores_only = predictions_df[MDmis_columns]
            predictions_df_scores_only.to_csv(os.path.join(result_dir, IDR_protein_start_end, "predictions.csv"))


def main():
    vault_dir = os.path.abspath(config["vault_dir"])
    data_dir = os.path.abspath(config["data_dir"])
    models_dir = os.path.abspath(config["models_dir"]) 
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--chunk", type = int, help="Divides the data into 10 chunks and uses this argument for which chunk to use.")
    args = parser.parse_args()

    MD_metadata = pd.read_csv(os.path.join(data_dir, "MD_metadata.csv"),
                                        index_col=0)

    MD_metadata.rename(columns={"source": "MD Data Source"}, inplace=True)
    IDRome_metadata = MD_metadata[MD_metadata["MD Data Source"] == "IDRome"]

    h5py_path = os.path.abspath(config["h5py_path"]) 

    aa_order = "ARNDCQEGHILKMFPSTWYV"
    aaindex_res_mat = np.load(os.path.join(data_dir, "aa_index1.npy"))
    res_data, pair_data = load_MD_data(h5py_path)


    ####

    MDmis_models = []
    for fold in range(1,6):
        MDmis = pickle.load(
            open(os.path.join(models_dir,
                               f"fold_{fold}",
                               "MDmis_RF_Res_Pair"), "rb")
        )
        MDmis_models.append(MDmis)


    chunks = np.array_split(IDRome_metadata, 10)
    
    generate_predictions(
            chunks[args.chunk], res_data, pair_data,
            aaindex_res_mat,
            aa_order,
            MDmis_models,
            os.path.join(vault_dir, "MDmis_predictions")
        )
        
        
    
if __name__ == "__main__":
    main()

import subprocess
import os
import re
import warnings
import tqdm
import glob
import pathlib
import numpy as np
import pandas as pd
import sys
ROOT = pathlib.Path(__file__).parents[1]
sys.path.append(str(ROOT))
from utils import *
from config import config

warnings.simplefilter(action='ignore', category=FutureWarning)



def compute_differences(res_feature_wt,
                        res_feature_mutated,
                        mutation_index):
    
    #First, compute local differences at the mutation location

    residue_mutation_difference = np.array(res_feature_mutated[mutation_index] - res_feature_wt[mutation_index])
    residue_mutation_ratio = np.array( (res_feature_mutated[mutation_index] + 1e-3) / (res_feature_wt[mutation_index] + 1e-3) )
    
    return (residue_mutation_difference, residue_mutation_ratio)




def main():
    data_dir = os.path.abspath(config["data_dir"])

    h5py_path = os.path.abspath(config["h5py_path"])
    res_data, pair_data = load_MD_data(h5py_path)
    
    wt_conformational_properties = pd.read_csv(os.path.join(
        data_dir, "conformational_properties.csv"
    ))

    res_md_diff_columns = [f'Res_MD_Diff_{i+1}' for i in range(47)] # number of Res MD features
    res_md_fc_columns = [f'Res_MD_FC_{i+1}' for i in range(47)] # number of Res MD features
    
    splitted_Long_IDRs = ["P02458_78_1254",
                            "O75179_1794_2603",
                            "P08123_1_1134",
                            "P29400_1_1456",
                            "Q01955_1_1441",
                            "Q12955_1434_2560",
                            "Q9UM47_1_1399",
                            "A2RUB1_1_749",
                            "Q6KC79_92_1196",
                            "P46100_289_1546"]

    mutation_differences_df = pd.DataFrame(columns= res_md_diff_columns + res_md_fc_columns + 
                                           ["Uniprot_ID", "Location"])
    
    processed_CALVADOS_directory = os.path.abspath(config["CALVADOS_processed_dir"])

    for split_LongIDR in tqdm.tqdm(splitted_Long_IDRs):
        split_LongIDR_uniprot= split_LongIDR.split("_")[0]


        # parse through the two folders, one for each splitted IDR. Then figure out which is first and which is second
        # then stitch their residue features together in order to have 1:1 comparison with the non-splitted simulations        
        res_features_list = []

        for folder in glob.glob(os.path.join(processed_CALVADOS_directory, 
                f"{split_LongIDR_uniprot}*")):
            
            split_folder_name = os.path.basename(folder)

            uniprot_id, start, end = split_folder_name.split("_")

            res_features_list.append((split_folder_name, uniprot_id, start, end))
        
        print(res_features_list)
        if res_features_list[0][2] < res_features_list[1][2]: # the first splitted IDR comes before the second

            res_feature_first = np.load(os.path.join(processed_CALVADOS_directory,
                res_features_list[0][0], "res_feature.npy"
            ))
            res_feature_second = np.load(os.path.join(processed_CALVADOS_directory,
                res_features_list[1][0], "res_feature.npy"
            ))

        else:
            res_feature_first = np.load(os.path.join(processed_CALVADOS_directory,
                res_features_list[1][0], "res_feature.npy"
            ))
            res_feature_second = np.load(os.path.join(processed_CALVADOS_directory,
                res_features_list[0][0], "res_feature.npy"
            ))
        
        res_features = np.concatenate([res_feature_first, res_feature_second], axis=0)
        print(res_features.shape)

        res_feature_wt = res_data[split_LongIDR]
        
        for i in range(res_feature_wt.shape[0]): #each index
            (residue_mutation_difference,
                residue_mutation_ratio) = compute_differences(res_feature_wt, res_features, i)
            
            mutation_differences_df.loc[len(mutation_differences_df.index)]= np.concatenate(
                (residue_mutation_difference, residue_mutation_ratio,
                np.array([split_LongIDR, i]) ),  
                axis = None
            )

    print(mutation_differences_df.head())

    mutation_differences_df.to_csv(
         os.path.join(data_dir, "split_LongIDRs_differences.csv")
    )
    
    
if __name__ == "__main__":
    main()

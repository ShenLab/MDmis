import subprocess
import os
import re
import warnings
import time
import argparse
import shutil
import glob

import numpy as np
import pandas as pd

from utils import *
warnings.simplefilter(action='ignore', category=FutureWarning)



def compute_differences(res_feature_wt, pair_feature_wt,
                        res_feature_mutated, pair_feature_mutated,
                        conformational_properties_wt, conformational_properties_mutated,
                        mutation_index):
    
    #First, compute local differences at the mutation location

    residue_mutation_difference = np.array(res_feature_mutated[mutation_index] - res_feature_wt[mutation_index])
    residue_mutation_ratio = np.array( (res_feature_mutated[mutation_index] + 1e-3) / (res_feature_wt[mutation_index] + 1e-3) )
    
    interaction_cutoff = 0.4
    pair_data_wt_above_cutoff = pair_feature_wt > interaction_cutoff
    count_pair_wt_md_features = (
        np.sum(pair_data_wt_above_cutoff[mutation_index, :, :9], axis=(0)) + 
        np.sum(pair_data_wt_above_cutoff[:, mutation_index, :9], axis=(0))   
        )
    
    pair_data_mut_above_cutoff = pair_feature_mutated > interaction_cutoff
    count_pair_mut_md_features = (
        np.sum(pair_data_mut_above_cutoff[mutation_index, :, :9], axis=(0)) + 
        np.sum(pair_data_mut_above_cutoff[:, mutation_index, :9], axis=(0))   
        )
    avg_pair_mutation_differences = np.concatenate((count_pair_mut_md_features- count_pair_wt_md_features,
                                                     np.mean(pair_data_mut_above_cutoff[mutation_index, :, 9]) - np.mean(pair_data_wt_above_cutoff[mutation_index, :, 9])),
                                                     axis = None)
    
    avg_pair_mutation_FC = np.concatenate((count_pair_mut_md_features+1e-3 / (count_pair_wt_md_features+1e-3),
                                                     np.mean(pair_data_mut_above_cutoff[mutation_index, :, 9]) / np.mean(pair_data_wt_above_cutoff[mutation_index, :, 9])),
                                                     axis = None)
    print(avg_pair_mutation_differences.shape)
    
    #avg_pair_mutation_ratio = np.array(pair_feature_mutated[mutation_index, :, :] + 1e-3 / pair_feature_wt[mutation_index, :, :] + 1e-3)
    #These should be same shape as our feature table (1,47) and (1,10), because they are location specific


    #Secondly, compute global differences 
    # print(np.linalg.norm(res_feature_mutated - res_feature_wt, axis = 1))
    # residue_euclid_distance = np.mean(np.linalg.norm(res_feature_mutated - res_feature_wt, axis = 1))
    # print(residue_euclid_distance)

    # average_residue_difference = np.mean(res_feature_mutated - res_feature_wt, axis = 0)


    #Thirdly, compute how conformational properties change
    Rg_difference = conformational_properties_mutated[0] - conformational_properties_wt["Rg"].values[0]
    Ete_difference = conformational_properties_mutated[1] - conformational_properties_wt["ete"].values[0]
    nu_difference = conformational_properties_mutated[2] - conformational_properties_wt["nu"].values[0]
    nu_FC = conformational_properties_mutated[2]/conformational_properties_wt["nu"].values[0]
    print(Rg_difference)
    print(Ete_difference)
    print(nu_difference)
    return (residue_mutation_difference, residue_mutation_ratio,
            avg_pair_mutation_differences, avg_pair_mutation_FC,
            Rg_difference, Ete_difference, nu_difference,
            nu_FC)




def main():
    data_dir = "/home/az2798/MDmis/data/"
    h5py_path = "/share/vault/Users/az2798/train_data_all/filtered_feature_all_ATLAS_GPCRmd_IDRome.h5"
    res_data, pair_data = load_MD_data(h5py_path)
    
    wt_conformational_properties = pd.read_csv(os.path.join(
        data_dir, "conformational_properties.csv"
    ))

    res_md_diff_columns = [f'Res_MD_Diff_{i+1}' for i in range(47)] # number of Res MD features
    res_md_fc_columns = [f'Res_MD_FC_{i+1}' for i in range(47)] # number of Res MD features
    pair_md_columns = [f'Avg_Pair_MD_Diff_{i+1}' for i in range(10)] # number of Pair MD features
    pair_md_fc_columns = [f'Avg_Pair_MD_FC_{i+1}' for i in range(10)] # number of Pair MD features

    #average_residue_difference_columns = [f'Res_All_MD_Diff_{i+1}' for i in range(47)] #number of Res MD features
    
    mutation_differences_df = pd.DataFrame(columns= res_md_diff_columns + res_md_fc_columns + 
                                           pair_md_columns +pair_md_fc_columns+
                                           ["Rg_Difference", 
                                            "Ete_Distance", "nu_Difference", "nu_FC",
                                            "Mutation_ID", "Mutation_Category"])
    
    processed_CALVADOS_directory =  "/nfs/user/Users/az2798/processed_CALVADOS/"

    for mutation_category_dir in glob.glob(os.path.join(processed_CALVADOS_directory, "*")):
        mutation_category = os.path.basename(mutation_category_dir)
        if mutation_category in ["Shorter_simulations", "Phosphorylations"]:
            continue
        for folder in glob.glob(os.path.join(mutation_category_dir, "*")):
            mutation_name = os.path.basename(folder)
            print(mutation_name)

            mutation_location = mutation_name.rsplit('_', 1)[1].split(':')[1]
            print(mutation_location)
            mutation_index = int(mutation_location) - int(mutation_name.split('_')[1])

            wt_name = mutation_name.rsplit('_', 1)[0]
            print(wt_name)

            region_length = int(wt_name.split('_')[-1]) - int(wt_name.split('_')[1]) + 1

            ###Renaming High RMSF, Low RMSF to Length Category
            if ("Pathogenic" in mutation_category) and (region_length > 800):
                mutation_category = "Pathogenic - Long IDRs"
            elif ("Pathogenic" in mutation_category) and (region_length <= 800):
                mutation_category = "Pathogenic - Short IDRs"

            res_feature_wt = res_data[wt_name]
            pair_feature_wt = pair_data[wt_name]

            print(pair_feature_wt.shape)
            res_feature_mutated = np.load(os.path.join(
                folder, "res_feature.npy"
            ))
            pair_feature_mutated = np.load(os.path.join(
                folder, "pair_feature.npy"
            ))
            conformational_properties_mutated = np.load(os.path.join(
                folder, "conformational_properties.npy"
            ))
            (residue_mutation_difference,
             residue_mutation_ratio,
                avg_pair_mutation_differences,
                avg_pair_mutation_FC,
                Rg_difference,
                Ete_difference,
                nu_difference,
                nu_FC) = compute_differences(
                                res_feature_wt, pair_feature_wt,
                                res_feature_mutated, pair_feature_mutated,
                                conformational_properties_wt=wt_conformational_properties[
                                    wt_conformational_properties["seq_name"] == wt_name],
                                conformational_properties_mutated=conformational_properties_mutated,
                                mutation_index = mutation_index)
            
            mutation_differences_df.loc[len(mutation_differences_df.index)]= np.concatenate(
                (residue_mutation_difference, residue_mutation_ratio,
                 avg_pair_mutation_differences,
                 avg_pair_mutation_FC,
                Rg_difference, Ete_difference, nu_difference,
                nu_FC,
                np.array([mutation_name, mutation_category])),  
                axis = None
            )


    print(mutation_differences_df.head())

    mutation_differences_df.to_csv(
         "/home/az2798/MDmis/data/calvados_mutation_differences.csv"
    )

    ############
    # shorter_runs_differences_df = pd.DataFrame(columns= res_md_diff_columns + res_md_fc_columns + 
    #                                         pair_md_columns + average_residue_difference_columns+
    #                                         ["Res_Euclid","Rg_Difference",
    #                                          "Ete_Distance", "Mutation_ID", "Mutation_Category"])
    
    # processed_CALVADOS_directory =  "/nfs/user/Users/az2798/processed_CALVADOS/"

    # for folder in glob.glob(os.path.join(processed_CALVADOS_directory,
    #                                       "Shorter_simulations", "*")):

    #     mutation_index = 0

    #     wt_name = os.path.basename(folder)
    #     print(wt_name)  

    #     res_feature_wt = res_data[wt_name]
    #     pair_feature_wt = pair_data[wt_name]

    #     print(pair_feature_wt.shape)
    #     res_feature_mutated = np.load(os.path.join(
    #         folder, "res_feature.npy"
    #     ))
    #     pair_feature_mutated = np.load(os.path.join(
    #         folder, "pair_feature.npy"
    #     ))
    #     conformational_properties_mutated = np.load(os.path.join(
    #         folder, "conformational_properties.npy"
    #     ))
    #     (residue_mutation_difference,
    #         residue_mutation_ratio,
    #         avg_pair_mutation_difference,
    #         average_residue_difference,
    #         residue_euclid_distance,
    #         Rg_difference,
    #         Ete_difference) = compute_differences(
    #                         res_feature_wt, pair_feature_wt,
    #                         res_feature_mutated, pair_feature_mutated,
    #                         conformational_properties_wt=wt_conformational_properties[
    #                             wt_conformational_properties["seq_name"] == wt_name],
    #                         conformational_properties_mutated=conformational_properties_mutated,
    #                         mutation_index = mutation_index)
        
    #     shorter_runs_differences_df.loc[len(shorter_runs_differences_df.index)]= np.concatenate(
    #         (residue_mutation_difference, residue_mutation_ratio,
    #             avg_pair_mutation_difference,
    #             average_residue_difference,
    #         residue_euclid_distance,
    #         Rg_difference, Ete_difference,
    #         np.array([wt_name, "Pathogenic_High_RMSF"])),  
    #         axis = None
    #     )


    # print(shorter_runs_differences_df.head())

    # shorter_runs_differences_df.to_csv(
    #     "/home/az2798/MDmis/data/shorter_runs_differences.csv"
    # )   

    
if __name__ == "__main__":
    main()

import cProfile
import math
import os
import pandas as pd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
import scipy.stats as ss

from utils import *
from predict_MDmis import *
from evaluate_MDmis import *
from compositional_analysis_AA import *



def main():
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 100)

    models_dir = "/home/az2798/MDmis/models/"
    data_dir = "/home/az2798/MDmis/data/"
    results_dir = "/home/az2798/MDmis/results/DMS_ROCs"
    vault_dir = "/share/vault/Users/az2798/"


    aa_order = "ARNDCQEGHILKMFPSTWYV"
    aaindex_res_mat = np.load(os.path.join(data_dir, "aa_index1.npy"))


    outcome_column_name = "outcome"
    #For DMS, let's use the entire data from any fold

    train_feature_table = pd.read_csv(
        os.path.join(data_dir, "DMS_train_val", "fold_1", "train.csv")
    )
    val_feature_table = pd.read_csv(
        os.path.join(data_dir, "DMS_train_val", "fold_1", "val.csv")
    )

    entire_feature_table = pd.concat([train_feature_table, val_feature_table])
    ####
    #ESM Zero shot scores
    ESM_table = pd.read_csv(os.path.join(vault_dir,
                                "ESM1b_data", "All_LLR_scores.csv"),
                                low_memory=False, index_col=0)
    
    
    #Merging to get ESM1b LLRs 
    entire_feature_table = pd.merge(left =entire_feature_table, right = ESM_table,
                          left_on= ["UniProtID", "location", "Changed AA"],
                          right_on=["Protein_ID", "Pos", "Changed_AA"], 
                          how = "left",
                          suffixes=["", "_y"]).drop(["Pos", "Original_AA", "Changed_AA",
                          "Protein_ID"], axis = 1)
    
    #entire_feature_table = val_feature_table
    print(entire_feature_table.shape)
    variant_information_columns = [outcome_column_name, "UniProtID", "location",
                                    "Original AA", 
                                    "Changed AA"]
    
    probability_table = entire_feature_table[variant_information_columns]

    use_model = "clinical"

    if use_model == "clinical":
        MDmis_AAindex = pickle.load(
            open(os.path.join(models_dir, "fold_1", "MDmis_RF_AAIndex"), "rb")
        )
        print(MDmis_AAindex.feature_names_in_)
        probability_table["MDmis_AAIndex_Scores"] = predict_MDmis(
            MDmis_AAindex, entire_feature_table, use_res_md = False,
             use_pair_md =  False, use_Cons= False, use_ESM_embed = False, use_conf_prop = False,
               regressor = False)

        MDmis_Res = pickle.load(
            open(os.path.join(models_dir, "fold_1", "MDmis_RF_Res"), "rb")
        )
        print(MDmis_Res.feature_names_in_)

        probability_table["MDmis_Res_Scores"] = predict_MDmis(
            MDmis_Res, entire_feature_table, use_res_md = True,
             use_pair_md =  False, use_Cons= False, use_ESM_embed = False, use_conf_prop = False, regressor = False)


        MDmis_Res_Pair = pickle.load(
            open(os.path.join(models_dir, "fold_1", "MDmis_RF_Res_Pair"), "rb")
        )
        probability_table["MDmis_Res_Pair_Scores"] = predict_MDmis(MDmis_Res_Pair, entire_feature_table,
                                                                    use_res_md = True,
             use_pair_md =  True, use_Cons= False, use_ESM_embed = False, use_conf_prop = False, regressor = False)
        
        # MDmis_ESM1b = pickle.load(
        #     open(os.path.join(models_dir, "fold_1", "MDmis_RF_ESM1b_MD"), "rb")
        # )
        # probability_table["MDmis_ESM1b_Scores"] = predict_MDmis(MDmis_ESM1b, entire_feature_table,
        #     use_res_md = True,
        #     use_pair_md =  True, use_Cons= True, use_ESM_embed = False,
        #      use_conf_prop = False, use_ESM1b = True, regressor = False)
    elif use_model == "DMS":
        MDmis_AAindex = pickle.load(
            open(os.path.join(models_dir, "MDmis_RF_AAIndex_DMS"), "rb")
        )
        probability_table["MDmis_AAIndex_Scores"] = predict_MDmis(
            MDmis_AAindex, entire_feature_table, False, False, regressor = True)

        MDmis_Res = pickle.load(
            open(os.path.join(models_dir, "MDmis_RF_Res_DMS"), "rb")
        )
        probability_table["MDmis_Res_Scores"] = predict_MDmis(
            MDmis_Res, entire_feature_table, True, False, regressor = True)


        MDmis_Res_Pair = pickle.load(
            open(os.path.join(models_dir, "MDmis_RF_Res_Pair_DMS"), "rb")
        )
        probability_table["MDmis_Res_Pair_Scores"] = predict_MDmis(MDmis_Res_Pair, entire_feature_table,
                                                                variant_information_columns, True, True,
                                                                regressor = True)

    else:
        MDmis_AAindex = pickle.load(
            open(os.path.join(models_dir, "MDmis_RF_AAIndex"), "rb")
        )
        clinical_AAIndex = predict_MDmis(
            MDmis_AAindex, entire_feature_table, False, False, regressor = False)
        MDmis_AAindex = pickle.load(
            open(os.path.join(models_dir, "MDmis_RF_AAIndex_DMS"), "rb")
        ) 
        DMS_AAIndex = predict_MDmis(
            MDmis_AAindex, entire_feature_table, False, False, regressor = True) 
        probability_table["MDmis_AAIndex_Scores"] = (clinical_AAIndex + DMS_AAIndex) /2 
        
        #########

        MDmis_Res = pickle.load(
            open(os.path.join(models_dir, "MDmis_RF_Res"), "rb")
        )
        clinical_Res = predict_MDmis(
            MDmis_Res, entire_feature_table, True, False, regressor = False)
        MDmis_Res = pickle.load(
            open(os.path.join(models_dir, "MDmis_RF_Res_DMS"), "rb")
        ) 
        DMS_Res = predict_MDmis(
            MDmis_Res, entire_feature_table, True, False, regressor = True) 
        probability_table["MDmis_Res_Scores"] = (clinical_Res + DMS_Res) /2 
        
        ##########

        MDmis_Res_Pair = pickle.load(
            open(os.path.join(models_dir, "MDmis_RF_Res_Pair"), "rb")
        )
        clinical_Res_Pair = predict_MDmis(
            MDmis_Res_Pair, entire_feature_table, True, True, regressor = False)
        
        MDmis_Res_Pair = pickle.load(
            open(os.path.join(models_dir, "MDmis_RF_Res_Pair_DMS"), "rb")
        ) 
        DMS_Res_Pair = predict_MDmis(
            MDmis_Res_Pair, entire_feature_table, True, True, regressor = True) 
        probability_table["MDmis_Res_Pair_Scores"] = (clinical_Res_Pair + DMS_Res_Pair) /2 
        #######

    ### Bringing in AlphaMissense Scores
    proteome_information = pd.read_csv(
        os.path.join(data_dir, "merged_proteome_information_DMS.csv"), index_col = 0,
        low_memory = False
    )

    proteome_information_subset = proteome_information[[
        "UniProtID", "location", "am_pathogenicity", "ESM_probabilities", "changed_aa_amis", "GERP++_RS",
        "DMS_score_bin", "DMS_score", "Assay_Type"
    ]]


    #proteome_information_subset.dropna(subset = ["am_pathogenicity", "GERP++_RS"], axis=0, inplace=True, ignore_index=True)
    print(entire_feature_table.shape)

    validation_set_information = pd.merge(entire_feature_table, 
             proteome_information_subset, 
             left_on = ["UniProtID","location", "Changed AA"],
             right_on = ["UniProtID", "location", "changed_aa_amis"],
             how = "inner")
    
    g = sns.FacetGrid(validation_set_information, col="DMS_score_bin", col_wrap=3, height=4, sharex=True, sharey=True)
    g.map(sns.histplot, "DMS_score", bins=30, color="blue", kde=False)
    g.set_titles("{col_name}")
    g.set_axis_labels("DMS Score", "Count")
    g.figure.suptitle("Histogram of DMS Scores by DMS Score Bin", y=1.05)
    g.tight_layout()
    plt.savefig(os.path.join(results_dir,
        "DMS_scores_by_label_IDR.png"), dpi=300
    )
    plt.clf()
    
    plt.clf()
    g = sns.FacetGrid(validation_set_information, col="Assay_Type", col_wrap=3, height=4, sharex=True, sharey=False)
    g.map(sns.histplot, "DMS_score", bins=30, color="blue", kde=False)
    g.set_titles("{col_name}")
    g.set_axis_labels("DMS Score", "Count")
    g.figure.suptitle("Histogram of DMS Scores by Assay Type", y=1.05)
    g.tight_layout()
    plt.savefig(os.path.join(results_dir,
        "DMS_scores_by_assay_IDR.png"), dpi=300
    )
    plt.clf()
    print(validation_set_information.shape)
    #print(validation_set_information)
    
    probability_table["AlphaMissense_Scores"] = validation_set_information["am_pathogenicity"]
    probability_table["ESM1b_Scores"] = validation_set_information["ESM_probabilities"]
    
    probability_table["Average_Amis_MDmis_Scores"] = probability_table[["AlphaMissense_Scores", 
                                                                        "MDmis_Res_Pair_Scores"]].mean(axis=1)
    print(validation_set_information["Assay_Type"].value_counts())
    plot_rhos_by_group(validation_set_information, 
              probability_table[["MDmis_AAIndex_Scores",
                                 "MDmis_Res_Scores",
                                 "MDmis_Res_Pair_Scores",
                                 "ESM1b_Scores",
                                 "AlphaMissense_Scores",
                                 "Average_Amis_MDmis_Scores"]].to_numpy().transpose(),
                                "DMS_score",
              "Assay_Type", "DMS (IDRs)", 
              ["MDmis (AAIndex)", 
               "MDmis (Res)",
               "MDmis (Res + Pair)",
               "ESM1b",
               "AlphaMissense",
               "AlphaMissense + MDmis (Res + Pair)"],
                os.path.join(results_dir, f"DMS_rhos_using_{use_model}_MDmis.png"))
    
    ##Positive control for AM
    proteome_information_subset.dropna(axis = 0, subset = ["am_pathogenicity"],
                                         inplace = True)
    print(ss.spearmanr(proteome_information_subset["am_pathogenicity"], 
                       proteome_information_subset["DMS_score"]) )
    rho_data = []
    for group in proteome_information_subset["Assay_Type"].unique():
        subset = proteome_information_subset[proteome_information_subset["Assay_Type"] == group]
        rho, p_val = ss.spearmanr(subset["DMS_score"], subset["am_pathogenicity"])
        rho_data.append({"Group": group, 'Spearman Rho': rho, 
                         'p-val': p_val, 'N': len(subset)})

    print(pd.DataFrame(rho_data), "Positive Control")

    proteome_information_subset.dropna(axis = 0, subset = ["ESM_probabilities"],
                                         inplace = True)
    print(ss.spearmanr(proteome_information_subset["ESM_probabilities"], 
                       proteome_information_subset["DMS_score"]) )
    rho_data = []
    for group in proteome_information_subset["Assay_Type"].unique():
        subset = proteome_information_subset[proteome_information_subset["Assay_Type"] == group]
        rho, p_val = ss.spearmanr(subset["DMS_score"], subset["ESM_probabilities"])
        rho_data.append({"Group": group, 'Spearman Rho': rho, 
                         'p-val': p_val, 'N': len(subset)})

    print(pd.DataFrame(rho_data), "Positive Control")




if __name__ == "__main__":
    main()
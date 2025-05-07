import cProfile
import math
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance

import warnings

warnings.filterwarnings("ignore")
import sys
import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.append(ROOT)
from utils import *
from config import config
from predict_MDmis import *
from evaluate_MDmis import *
    

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
def main():
    models_dir = os.path.abspath(config["models_dir"])
    data_dir = os.path.abspath(config["data_dir"]) 
    results_dir = os.path.abspath(config["results_dir"])

    val_tables = []
    for fold in range(1,6):

        val_feature_table = pd.read_csv(
            os.path.join(data_dir, "clinical_train_val",
                         f"fold_{fold}", "val.csv")
        )
        val_feature_table["Fold"] = fold
        val_tables.append(val_feature_table)
    

    
    IDR_MD_features = pd.concat(val_tables, 
                                axis=0)
    IDR_MD_features = IDR_MD_features[IDR_MD_features["Label Source"]!= "PrimateAI"] # remove the PrimateAI benign variants
    print(IDR_MD_features.shape)

    
    #IDR_MD_features.drop(columns = ["Changed_AA_y", "Original_AA_y"], inplace=True)
    proteome_information = pd.read_csv(
        os.path.join(data_dir, "merged_proteome_information_clinical.csv"),
        index_col = 0, low_memory= False
    )
    location_column_name = "location"
    outcome_column_name = "outcome"
    IDRs_table = pd.merge(IDR_MD_features, 
                          proteome_information,
                          left_on=["UniProtID", location_column_name, "Changed AA"],
                          right_on=["UniProtID", location_column_name, "changed_aa_amis"],
                          how="left", suffixes=('', '_y')).drop(["LLR_y"], axis = 1)
    


    ## Complete merged data with proteome information and MD features 
    predictions_labels_dict = {
        "Length_Scores": "IDR Length",
        "MDmis_AAIndex_Scores": "AAindex",
        "MDmis_Res_Scores": "MDmis (Res)",
        "MDmis_Res_Pair_Scores": "MDmis (Res and Pair)",
        "am_pathogenicity": "AlphaMissense",
        "ESM_probabilities": "ESM1b",
        "ESM1b_MDmis_Scores": "MDmis + MSA + ESM1b"
        
    }   

    MDmis_models = []
    MDmis_ESM_models = []
    for fold in range(1,6):
        OneHot = pickle.load(
            open(os.path.join(models_dir,
                                f"fold_{fold}",
                                "MDmis_RF_One_Hot"), "rb")
        )
        IDRs_table.loc[IDRs_table["Fold"] == fold, "MDmis_OH_Scores"] = predict_MDmis(
            OneHot, IDRs_table[IDRs_table["Fold"] == fold], use_OH=True, use_AA_index = False,
            use_res_md = False,
             use_pair_md =  False, use_Cons= False, use_ESM_embed = False, use_conf_prop = False)
        
        MDmis_AAindex = pickle.load(
            open(os.path.join(models_dir,
                                f"fold_{fold}",
                                "MDmis_RF_AAIndex"), "rb")
        )
        IDRs_table.loc[IDRs_table["Fold"] == fold, "MDmis_AAIndex_Scores"] = predict_MDmis(
            MDmis_AAindex, IDRs_table[IDRs_table["Fold"] == fold],
            use_res_md = False,
             use_pair_md =  False, use_Cons= False, use_ESM_embed = False, use_conf_prop = False)

        MDmis_Res = pickle.load(
            open(os.path.join(models_dir,
                               f"fold_{fold}",
                               "MDmis_RF_Res"), "rb")
        )
        IDRs_table.loc[IDRs_table["Fold"] == fold, "MDmis_Res_Scores"] = predict_MDmis(
            MDmis_Res, IDRs_table[IDRs_table["Fold"] == fold],
            use_res_md = True,
             use_pair_md =  False, use_Cons= False, use_ESM_embed = False, use_conf_prop = False)


        MDmis_Res_Pair = pickle.load(
            open(os.path.join(models_dir,
                               f"fold_{fold}",
                               "MDmis_RF_Res_Pair"), "rb")
        )
        IDRs_table.loc[IDRs_table["Fold"] == fold, "MDmis_Res_Pair_Scores"] = predict_MDmis(
            MDmis_Res_Pair, IDRs_table[IDRs_table["Fold"] == fold],
            use_res_md = True,
             use_pair_md =  True, use_Cons= False, use_ESM_embed = False, use_conf_prop = False)
        
        MDmis_models.append(MDmis_Res_Pair)

        ESM1b_MDmis_only = pickle.load(
            open(os.path.join(models_dir,
                               f"fold_{fold}",
                               "MDmis_RF_ESM1b_MD_only"), "rb")
        )
        IDRs_table.loc[IDRs_table["Fold"] == fold, "ESM1b_MDmis_only_Scores"] = predict_MDmis(
            ESM1b_MDmis_only, IDRs_table[IDRs_table["Fold"] == fold], use_res_md= True,
             use_pair_md= True, use_Cons =False, use_ESM_embed= False, use_conf_prop = False,
             use_ESM1b=True)
        
        ESM1b_MDmis = pickle.load(
            open(os.path.join(models_dir,
                               f"fold_{fold}",
                               "MDmis_RF_ESM1b_MD_Cons"), "rb")
        )
        IDRs_table.loc[IDRs_table["Fold"] == fold, "ESM1b_MDmis_Scores"] = predict_MDmis(
            ESM1b_MDmis, IDRs_table[IDRs_table["Fold"] == fold], use_res_md= True,
             use_pair_md= True, use_Cons =True, use_ESM_embed= False, use_conf_prop = False,
             use_ESM1b=True)
        
        MDmis_ESM_models.append(ESM1b_MDmis)

        Length_model = pickle.load(
            open(os.path.join(models_dir,
                               f"fold_{fold}",
                               "MDmis_RF_Length"), "rb")
        )
        IDRs_table.loc[IDRs_table["Fold"] == fold, "Length_Scores"] = predict_MDmis(
            Length_model, IDRs_table[IDRs_table["Fold"] == fold], use_res_md= False,
             use_pair_md= False, use_Cons =False, use_ESM_embed= False, use_conf_prop = False,
             use_ESM1b=False, use_length = True)
        

    
    #print(IDRs_table.head)
    IDRs_table.dropna(subset = ["am_pathogenicity", "GERP++_RS"], axis = 0,
                                      inplace=True)

    # IDRs_table["Average_Amis_MDmis_Scores"] = IDRs_table[["am_pathogenicity", 
    #                                                                     "MDmis_Res_Pair_Scores"]].mean(axis=1)
        
        
    ### Plotting entire validation set
    GERP_column_name = "GERP++_RS"
    GERP_cutoff = 2
    
    IDRs_table["start"] = IDRs_table["protein_start_end"].str.split("_").str[1].astype(int)
    IDRs_table["end"] = IDRs_table["protein_start_end"].str.split("_").str[2].astype(int)

    IDRs_table["Region Length"] = IDRs_table["end"] - IDRs_table["start"] + 1

    IDRs_table["GERP_Category"] = np.select([
        IDRs_table[GERP_column_name] > GERP_cutoff, 
        IDRs_table[GERP_column_name] <= GERP_cutoff
    ],
    ["High Constraint", "Low Constraint"])
    IDRs_table["Length_Category"] = np.select([
        IDRs_table["Region Length"] > 800, 
        IDRs_table["Region Length"] <= 800
    ],
    ["Long IDRs", "Short IDRs"])
    plot_average_roc(IDRs_table,
                     predictions_labels_dict,
                     5,
                     "Fold",
                     outcome_column_name, 0.1, "Validation performance for entire dataset", 
                     os.path.join(results_dir, "Clinical_ROCs",
                                  "ROC_entire_validation_0.1.png"))
    plot_average_roc(IDRs_table,
                     predictions_labels_dict,
                     5,
                     "Fold",
                     outcome_column_name,None, "Validation performance for entire dataset", 
                     os.path.join(results_dir, "Clinical_ROCs",
                                  "ROC_entire_validation.png"))
    plot_average_roc(IDRs_table[IDRs_table["Length_Category"] == "Long IDRs"],
                     predictions_labels_dict,
                     5,
                     "Fold",
                     outcome_column_name, None, "Validation performance for Long IDRs", 
                     os.path.join(results_dir, "Clinical_ROCs",
                                  "ROC_longIDRs.png"))
    
    plot_average_roc(IDRs_table[IDRs_table["Length_Category"] == "Long IDRs"],
                     predictions_labels_dict,
                     5,
                     "Fold",
                     outcome_column_name, 0.05, "Validation performance for Long IDRs", 
                     os.path.join(results_dir, "Clinical_ROCs",
                                  "ROC_longIDRs_0.05.png"))
    plot_average_roc(IDRs_table[IDRs_table["Length_Category"] == "Long IDRs"],
                     predictions_labels_dict,
                     5,
                     "Fold",
                     outcome_column_name, 0.1, "Validation performance for Long IDRs", 
                     os.path.join(results_dir, "Clinical_ROCs",
                                  "ROC_longIDRs_0.1.png"))
    plot_average_roc(IDRs_table[IDRs_table["Length_Category"] == "Short IDRs"],
                     predictions_labels_dict,
                     5,
                     "Fold",
                     outcome_column_name, None, "Validation performance for Short IDRs", 
                     os.path.join(results_dir, "Clinical_ROCs",
                                  "ROC_shortIDRs.png"))
    plot_average_roc(IDRs_table[IDRs_table["Length_Category"] == "Short IDRs"],
                     predictions_labels_dict,
                     5,
                     "Fold",
                     outcome_column_name, 0.1, "Validation performance for Short IDRs", 
                     os.path.join(results_dir, "Clinical_ROCs",
                                  "ROC_shortIDRs_0.1.png"))
    
    plot_average_roc(IDRs_table[IDRs_table["GERP_Category"] == "High Constraint"],
                     predictions_labels_dict,
                     5,
                     "Fold",
                     outcome_column_name, 0.1, "Validation performance for highly conserved sites", 
                     os.path.join(results_dir, "Clinical_ROCs",
                                  "ROC_high_constraint_0.1.png"))
    plot_average_roc(IDRs_table[IDRs_table["GERP_Category"] == "Low Constraint"],
                     predictions_labels_dict,
                     5,
                     "Fold",
                     outcome_column_name, 0.1, "Validation performance for poorly conserved sites", 
                     os.path.join(results_dir, "Clinical_ROCs",
                                  "ROC_low_constraint_0.1.png"))


    feature_dict = {"Res_MD_3": "RMSF",
                    "Res_MD_2": "Standard Deviation\nSASA",
                    "Res_MD_1": "Average\nSASA",
                    "Pair_MD_10": "Average Covariance\nof Residue",
                    "Res_MD_34": "Proportion of Phi Angle\n11th Percentile"}
    feature_importance_MDmis(MDmis_models,
                             IDRs_table,outcome_column_name,
                             os.path.join(results_dir, "Clinical_ROCs",
                                  "feature_importances_MDmis.png"),
                            feature_dict, num_features = 10,
             use_res_md= True,
             use_pair_md= True, use_Cons =False, use_ESM_embed= False,
             use_conf_prop = False,
             use_ESM1b=False)

    feature_importance_MDmis(MDmis_ESM_models,
                             IDRs_table,outcome_column_name,
                             os.path.join(results_dir, "Clinical_ROCs",
                                  "feature_importances_MDmis_ESM1b.png"),
                            feature_dict,num_features = 10,
             use_res_md= True,
             use_pair_md= True, use_Cons =True, use_ESM_embed= False,
             use_conf_prop = False,
             use_ESM1b=True)

if __name__ == "__main__":
    main()

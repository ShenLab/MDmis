
import os
import pandas as pd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pathlib
import sys
ROOT = pathlib.Path(__file__).parent
sys.path.append(ROOT)
from utils import *
from config import config


def train_MDmis_RF(train_feature_table, outcome_column_name,
                   use_OH =False,
                use_AA_index = True,
                use_res_md = True, use_pair_md = True,
                use_Cons = True,
                use_ESM_embed = True, 
                use_conf_prop = False,
                use_ESM1b = False,
                use_length = False,
                store_model = True,
                models_directory = None,
                model_suffix = None,
                fold = None):
    selected_features = []
    if use_OH:
        selected_features.append(train_feature_table.columns[train_feature_table.columns.str.contains('One_Hot_')])
    if use_AA_index:
        selected_features.append(train_feature_table.columns[train_feature_table.columns.str.contains('Res_AA_')])
    if use_res_md:
        selected_features.append(train_feature_table.columns[train_feature_table.columns.str.contains('Res_MD')]) #selecting residue features
    if use_pair_md:
        selected_features.append(train_feature_table.columns[train_feature_table.columns.str.contains('Pair_MD')]) #selecting pair features
    if use_Cons:
        selected_features.append(train_feature_table.columns[train_feature_table.columns.str.contains('Cons_')]) #selecting Cons features
    if use_ESM_embed:
        selected_features.append(train_feature_table.columns[train_feature_table.columns.str.contains('ESM2_')]) #selecting ESM2 features
    
    if use_conf_prop:
        selected_features.append(train_feature_table.columns[train_feature_table.columns.str.contains("Conf_")]) #selecting conformational properties
    if use_ESM1b:
        selected_features.append(train_feature_table.columns[train_feature_table.columns.str.contains("ESM1b_")]) #selecting ESM1b LLR
    if use_length:
        selected_features.append(train_feature_table.columns[train_feature_table.columns.str.contains("Region Length")]) #selecting Length of region for benchmark

    #rus = RandomUnderSampler(random_state=19, sampling_strategy = 0.8)

    #training the model
    X_train = train_feature_table.loc[:, pd.Index(np.concatenate(selected_features))]
    MDmis_RF = RandomForestClassifier(random_state=42, max_depth=15
                                    )
    y_train = train_feature_table[outcome_column_name].astype(int)

    MDmis_RF.fit(X_train, y_train)

    if store_model:
        if model_suffix == None or models_directory == None:
            raise ValueError("Model suffix and directory cannot be NoneType.")
        else:   
            model_path = os.path.join(models_directory,
                                      f"fold_{fold}",
                                      f"MDmis_RF_{model_suffix}")
            os.makedirs(os.path.join(models_directory,
                                      f"fold_{fold}"),
                                        exist_ok=True) #make the parent directory

            with open(model_path, "wb") as f:
                pickle.dump(MDmis_RF, f)
    

def train_MDmis_regression(train_feature_table, variant_information_columns, outcome_column_name,
                use_res_md = True, use_pair_md = True, store_model = True,
                models_directory = None,
                model_suffix = None):
    
    if not use_res_md:
        train_feature_table = train_feature_table.loc[:,~train_feature_table.columns.str.contains('Res_MD')] #removing residue features
    if not use_pair_md:
        train_feature_table = train_feature_table.loc[:,~train_feature_table.columns.str.contains('Pair_MD')] #removing pair features

    #training the model

    MDmis_RF = RandomForestRegressor(random_state=13)
    X_train = train_feature_table.drop(columns=variant_information_columns)
    y_train = train_feature_table[outcome_column_name].astype(int)
    MDmis_RF.fit(X_train, y_train)

    if store_model:
        if model_suffix == None or models_directory == None:
            raise ValueError("Model suffix and directory cannot be NoneType.")
        else:   
            model_path = os.path.join(models_directory, f"MDmis_RF_{model_suffix}")
            with open(model_path, "wb") as f:
                pickle.dump(MDmis_RF, f)


def split_benign(benign_df, num_splits):
    shuffled_indices = np.random.permutation(benign_df.index)
    df_size = len(benign_df)//num_splits
    remainder = len(benign_df) % num_splits
    benign_dfs = []
    start_index = 0
    for k in range(num_splits):
        end_index = start_index + df_size + (1 if k < remainder else 0)
        current_indices = shuffled_indices[start_index:end_index]
        benign_dfs.append(benign_df.loc[current_indices])
        start_index = end_index
    return benign_dfs


def create_folds(feature_table, group_to_split,
                 data_dir):
    clinvar_benign = feature_table[(feature_table["Label Source"] == "ClinVar") & (feature_table["outcome"] == 0)]
    primateai_benign = feature_table[(feature_table["Label Source"] == "PrimateAI") & (feature_table["outcome"] == 0)]
    pathogenic = feature_table.drop(clinvar_benign.index).drop(primateai_benign.index)

    unique_proteins_df = pd.DataFrame(pathogenic[group_to_split].unique(),
                                       columns=[group_to_split])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print(unique_proteins_df.shape)

    ## To shuffle the benign variants in testing
    bening_val_dfs = split_benign(clinvar_benign,5)

    #Iterate through each fold
    for fold_idx, (train_index, val_index) in enumerate(kf.split(unique_proteins_df)):
        train_proteins = unique_proteins_df.iloc[train_index]
        val_proteins = unique_proteins_df.iloc[val_index]

        print(val_proteins)
        
        train_feature_table = pathogenic[pathogenic[group_to_split].isin(train_proteins[group_to_split])]
        val_feature_table = pathogenic[pathogenic[group_to_split].isin(val_proteins[group_to_split])]

        ## Adding the Benign Variants back into train and val
        train_feature_table = pd.concat([train_feature_table, primateai_benign], ignore_index=True)
        val_feature_table = pd.concat([val_feature_table, bening_val_dfs[fold_idx]], ignore_index=True)

        #print(train_feature_table.shape)
        #print(val_feature_table.shape)
        fold_dir = os.path.join(data_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        train_feature_table.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_feature_table.to_csv(os.path.join(fold_dir, "val.csv"), index=False)

def train_one_fold(data_dir, models_dir,
                    fold):
    train_feature_table = pd.read_csv(
            os.path.join(data_dir, "clinical_train_val",
                         f"fold_{fold}", "train.csv")
        )
    
    train_MDmis_RF(train_feature_table,
                                "outcome", 
                                use_OH=True,
                                use_AA_index=False,
                                use_res_md= False,
                                use_pair_md= False,
                                use_Cons=False,
                                use_ESM_embed=False,
                                use_conf_prop=False,
                                store_model=True, 
                                models_directory=models_dir, 
                                model_suffix="One_Hot",
                                fold = fold)
    
    train_MDmis_RF(train_feature_table,
                                "outcome", 
                                use_res_md= False,
                                use_pair_md= False,
                                use_Cons=False,
                                use_ESM_embed=False,
                                use_conf_prop=False,
                                store_model=True, 
                                models_directory=models_dir, 
                                model_suffix="AAIndex",
                                fold = fold)
    
    train_MDmis_RF(train_feature_table,
                                "outcome", use_res_md=True, 
                                use_pair_md=False,
                                use_Cons=False,
                                use_ESM_embed=False,
                                use_conf_prop=False,
                                store_model= True, 
                                models_directory= models_dir, 
                                model_suffix= "Res", 
                                fold = fold)
    train_MDmis_RF(train_feature_table,
                                "outcome", 
                                use_res_md = True, 
                                use_pair_md=True,
                                use_Cons=False,
                                use_ESM_embed= False,
                                use_conf_prop=False,
                                store_model= True, 
                                models_directory=models_dir, 
                                model_suffix="Res_Pair",
                                fold = fold)
    train_MDmis_RF(train_feature_table,
                                "outcome", 
                                use_res_md = False, 
                                use_pair_md=False, 
                                use_Cons=False,
                                use_ESM_embed= False,
                                use_ESM1b=True,
                                use_conf_prop=False,
                                store_model= True, 
                                models_directory=models_dir, 
                                model_suffix="ESM1b",
                                fold = fold)
    train_MDmis_RF(train_feature_table,
                                "outcome",
                                use_AA_index=True, 
                                use_res_md = True, 
                                use_pair_md=True,
                                use_Cons=True,
                                use_ESM_embed= False,
                                use_conf_prop=False,
                                store_model= True, 
                                models_directory=models_dir, 
                                model_suffix="MD_Cons",
                                fold = fold)
    train_MDmis_RF(train_feature_table,
                                "outcome", 
                                use_res_md = True, 
                                use_pair_md=True,
                                use_Cons=False,
                                use_ESM_embed= False,
                                use_conf_prop=False,
                                use_ESM1b=True,
                                store_model= True, 
                                models_directory=models_dir, 
                                model_suffix="ESM1b_MD_only",
                                fold = fold)
    train_MDmis_RF(train_feature_table,
                                "outcome", 
                                use_res_md = False, 
                                use_pair_md=False,
                                use_Cons=False,
                                use_ESM_embed= False,
                                use_conf_prop=False,
                                use_ESM1b=False,
                                use_length=True,
                                store_model= True, 
                                models_directory=models_dir, 
                                model_suffix="Length",
                                fold = fold)
    train_MDmis_RF(train_feature_table,
                                "outcome", 
                                use_res_md = True, 
                                use_pair_md=True,
                                use_Cons=True,
                                use_ESM_embed= False,
                                use_ESM1b=True,
                                use_conf_prop=False,
                                store_model= True, 
                                models_directory=models_dir, 
                                model_suffix="ESM1b_MD_Cons",
                                fold = fold)
    

def main():
    data_dir = os.path.abspath(config["data_dir"])
    models_dir = os.path.abspath(config["models_dir"])
    vault_dir = os.path.abspath(config["vault_dir"])

    entire_feature_table = pd.read_csv(
        os.path.join(data_dir, "clinical_train_val", "feature_table.csv"
    )
    )
    entire_feature_table = entire_feature_table[entire_feature_table["Label Source"]!= "HGMD"] # not used for training

 
    print(entire_feature_table.head())
    sequence_composition_df = pd.read_csv(
        os.path.join(data_dir, "sequence_composition_MSA.csv"), index_col = 0
    )
    print(sequence_composition_df.shape, "Before")

    sequence_composition_df.drop_duplicates(subset=["UniProtID", "Location"],inplace=True)
    print(sequence_composition_df.shape, "After")
    sequence_composition_df.rename(columns={
        "Average Cosine Similarity": "Cons_1",
        "Average Entropy Region": "Cons_2",
        "Average Entropy Site": "Cons_3",
        "Average Gamma": "Cons_4",
        "StDev Gamma": "Cons_5"
    }, inplace=True)

    feature_table_with_composition = pd.merge(left = entire_feature_table, right = sequence_composition_df,
                                           left_on= ["UniProtID", "location"],
                                           right_on= ["UniProtID", "Location"],
                                           how = "inner")
    feature_table_with_composition.drop("Location", axis=1,
     inplace = True)
    print(feature_table_with_composition.head())
    
    
    conformational_properties = pd.read_csv(os.path.join(
        data_dir, "conformational_properties.csv"
    ))
    conformational_properties = conformational_properties[["seq_name", "ete", "nu"]]
    conformational_properties.rename(columns = {"ete": "Conf_1", "nu": "Conf_2"}, inplace=True)
    
    feature_table_with_conf = pd.merge(left = feature_table_with_composition, right = conformational_properties,
                                           left_on= "protein_start_end",
                                           right_on="seq_name",
                                           how="left")

    ESM_table = pd.read_csv(os.path.join(vault_dir,
                                "ESM1b_data", "All_LLR_scores.csv"),
                                low_memory=False, index_col=0)
    scaler = MinMaxScaler()
    ESM_table["ESM1b_probabilities"] = scaler.fit_transform(-1*ESM_table[['LLR']])
    feature_table_with_ESM1b = pd.merge(left =feature_table_with_conf, right = ESM_table,
                          left_on= ["UniProtID", "location", "Changed AA"],
                          right_on=["Protein_ID", "Pos", "Changed_AA"], 
                          how = "left",
                          suffixes=["", "_y"]).drop(["Pos", "Original_AA", "Changed_AA",
                          "Protein_ID"], axis = 1)
    
    print(feature_table_with_ESM1b.head())
    feature_table_with_ESM1b["start"] = feature_table_with_ESM1b["protein_start_end"].str.split("_").str[1].astype(int)
    feature_table_with_ESM1b["end"] = feature_table_with_ESM1b["protein_start_end"].str.split("_").str[2].astype(int)

    feature_table_with_ESM1b["Region Length"] = feature_table_with_ESM1b["end"] - feature_table_with_ESM1b["start"] + 1

    print("Feature Table for Splits", feature_table_with_ESM1b.shape)

    create_folds(feature_table_with_ESM1b, "UniProtID",
                 os.path.join(data_dir, "clinical_train_val"))
    


    ####
    #run folds in Paralles
    Parallel(n_jobs=6)(delayed(train_one_fold)(data_dir, models_dir, 
                                            fold) for fold in range(1, 6))
        

    train_feature_table = pd.read_csv(
        os.path.join(data_dir, "DMS_train_val", "train.csv"), index_col=0
    )
    variant_information_columns = ["DMS_z_score", "UniProtID", "Location",
                                    "Original AA", 
                                    "Changed AA"]
    train_MDmis_regression(train_feature_table, variant_information_columns,
                                   "DMS_z_score", False, False, True, 
                                   models_dir, "AAIndex_DMS")
    
    train_MDmis_regression(train_feature_table, variant_information_columns,
                                   "DMS_z_score", True, False, True, 
                                   models_dir, "Res_DMS")
    train_MDmis_regression(train_feature_table, variant_information_columns,
                                   "DMS_z_score", True, True, True, 
                                   models_dir, "Res_Pair_DMS")

if __name__ == "__main__":
    main()
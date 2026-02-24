import os
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pathlib
import sys
ROOT = pathlib.Path(__file__).parents[1]
sys.path.append(str(ROOT))
from utils import *
from config import config

from train_MDmis_RF import train_MDmis_RF
from predict_MDmis import predict_MDmis

def main():
    data_dir = os.path.abspath(config["data_dir"])
    models_dir = os.path.abspath(config["models_dir"])
    vault_dir = os.path.abspath(config["vault_dir"])

    train_feature_table = pd.read_csv(
        os.path.join(data_dir, "clinical_train_val", "fold_1", "train.csv")
    )
    val_feature_table = pd.read_csv(
            os.path.join(data_dir, "clinical_train_val", "fold_1", "val.csv")
        )
    outcome_column_name = "outcome"
    
    res_md_cols = train_feature_table.columns[
        train_feature_table.columns.str.startswith("Res_MD_")
    ]
    res_md_features = (
        pd.Series(res_md_cols)
        .str.extract(r"Res_MD_(\d+)_pos_")[0]
    )
    res_md_groups = {
        f"Res_MD_{feat}": res_md_cols[res_md_features == feat].to_numpy()
        for feat in res_md_features.unique()
    }


    pair_md_groups = {
        col: np.array([col])
        for col in train_feature_table.columns
        if col.startswith("Pair_MD_")
    }
    cons_groups = {
        col: np.array([col])
        for col in train_feature_table.columns
        if col.startswith("Cons_")
    }

    esm_groups = {
    col: np.array([col])
        for col in train_feature_table.columns
        if col.startswith("ESM1b_")
    }

    
    features_to_remove = {**res_md_groups, **pair_md_groups, 
                          **cons_groups, **esm_groups}
    

    MDmis_all = train_MDmis_RF(train_feature_table,
                                "outcome", 
                                use_res_md = True, 
                                use_pair_md=True,
                                use_Cons=True,
                                use_ESM_embed= False,
                                use_ESM1b=True,
                                use_conf_prop=False,
                                store_model= False, 
                                models_directory=None, 
                                model_suffix="None",
                                fold = 1,
                                feature_to_remove = None)
    probs_all = predict_MDmis(
        MDmis_all, val_feature_table,
        use_AA_index=True,
        use_res_md=True,
        use_pair_md=True, use_Cons=True,
        use_ESM1b=True,
        feature_to_remove=None
    )
    baseline_performance = roc_auc_score(val_feature_table[outcome_column_name],
                                         probs_all)
    
    non_baseline_performances ={}
    for feature_name, feature_cols in tqdm(features_to_remove.items()):
        print(feature_name   , "Removed Feature")
        MDmis_ablation = train_MDmis_RF(train_feature_table,
                                "outcome", 
                                use_res_md = True, 
                                use_pair_md=True,
                                use_Cons=True,
                                use_ESM_embed= False,
                                use_ESM1b=True,
                                use_conf_prop=False,
                                store_model= False, 
                                models_directory=None, 
                                model_suffix="None",
                                fold = 1,
                                feature_to_remove = feature_cols)
        
        probs = predict_MDmis(
            MDmis_ablation, val_feature_table,
            use_AA_index=True,
            use_res_md=True,
            use_pair_md=True, use_Cons=True,
            use_ESM1b=True,
            feature_to_remove=feature_cols
        )

        non_baseline_performances[feature_name] = roc_auc_score(val_feature_table[outcome_column_name],
                                                                probs) - baseline_performance
    non_baseline_performances = dict(sorted(non_baseline_performances.items(), key=lambda item: item[1]))
    print(non_baseline_performances)

    


if __name__ == "__main__":
    main()
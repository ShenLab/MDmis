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

def predict_MDmis(MDmis_model, val_feature_table, 
                  use_OH = False,
                  use_AA_index= True,
                  use_res_md = True, use_pair_md = True,
                  use_Cons = True,
                  use_ESM_embed = False,
                  use_conf_prop = False,
                  use_ESM1b = False,
                  use_length = False,
                  regressor = False):
    
    """
    Predicts pathogenicity probabilities for variants using the given MDmis model, 
    based on molecular dynamics (MD) and variant features.

    Parameters:
    ----------
    MDmis_model : sklearn-like model
        The pre-trained model that can take variant feature tables as input and predict probabilities.
        
    val_feature_table : pandas.DataFrame
        A DataFrame containing the validation features, including molecular dynamics (MD) data and 
        variant-specific information.

    use_res_md : bool, optional
        A flag indicating whether to include residue-level MD features in the prediction. Default is True.
        
    use_pair_md : bool, optional
        A flag indicating whether to include pairwise MD features in the prediction. Default is True.
    use_Cons : bool, optional
        A flag indicating whether to include sequence conservation properties in the prediction. Default is True.
    use_ESM_embed : bool, optional
        A flag indicating whether to include ESM embeddings in the prediction. Default is True.
    use_conf_prop : bool, optional
        A flag indicating whether to include conformational properties of the IDR in the prediction. Default is False.
    use_ESM1b : bool, optional
        A flag indicating whether to include ESM1b zero-shot score. Default is False.    

    use_length : bool, optional
        A flag indicating whether to include length of the IDR in the prediction. Default is False.    

    
    regressor : bool, optional
        A flag to indicate whether this is a regressor or not
    Returns:
    --------
    y_prob : numpy.ndarray
        An array containing the predicted probabilities of pathogenicity for each variant in the validation set.
    """
    
    selected_features = []
    if use_OH:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('One_Hot_')])
    if use_AA_index:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('Res_AA_')])
    if use_res_md:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('Res_MD')])
    if use_pair_md:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('Pair_MD')])
    if use_Cons:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('Cons_')])
    if use_ESM_embed:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('ESM2_')])
    if use_conf_prop:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('Conf_')])
    if use_ESM1b:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains("ESM1b_probabilities")]) #selecting ESM1b LLR
    if use_length:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains("Region Length")]) #selecting Length of region for benchmark


    #Combine selected features into a single DataFrame
    X_val = val_feature_table.loc[:, pd.Index(np.concatenate(selected_features))]

    #Make predictions using the model
    if regressor:
        y_prob = MDmis_model.predict(X_val)
    else:
        y_prob = MDmis_model.predict_proba(X_val)[:, 1]

    return y_prob


def feature_importance_MDmis(MDmis_model, val_feature_table,
                            outcome_column_name,
                            plot_filename,
                            feature_dict, permutation=False,
                            num_features = 5,
                  use_res_md = True, use_pair_md = True,
                  use_Cons = True,
                  use_ESM_embed = True,
                  use_conf_prop = False,
                  use_ESM1b = False,
                  use_length = False):
    selected_features = []
    selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('Res_AA_')])
    if use_res_md:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('Res_MD')])
    if use_pair_md:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('Pair_MD')])
    if use_Cons:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('Cons_')])
    if use_ESM_embed:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('ESM2_')])
    if use_conf_prop:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains('Conf_')])
    if use_ESM1b:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains("LLR")]) #selecting ESM1b LLR
    if use_length:
        selected_features.append(val_feature_table.columns[val_feature_table.columns.str.contains("Region Length")]) #selecting Length of region for benchmark

    X_val = val_feature_table.loc[:, pd.Index(np.concatenate(selected_features))]
    plt.figure(figsize=(8,6))
    ax =plt.gca()
    if permutation:
        if isinstance(MDmis_model, list) and len(MDmis_model) > 1:
            all_importances = []

            for model in MDmis_model:
                result = permutation_importance(
                    model, X_val, val_feature_table[outcome_column_name], n_repeats=10,
                    random_state=21
                )
                all_importances.append(result.importances_mean)

            importance_df = pd.DataFrame(all_importances, columns=X_val.columns)
            importance_df.rename(columns = feature_dict, inplace=True)
            avg_importances = importance_df.mean(axis=0)
            std_importances = importance_df.std(axis=0, ddof=1)
            avg_importances.sort_values(ascending=False, inplace=True)
            std_importances = std_importances.loc[avg_importances.index]
            print(avg_importances.head(num_features))

            avg_importances.head(num_features).plot.barh(xerr=std_importances.head(num_features), ax=ax, capsize=5, color='blue', alpha=0.75)
            plt.xlabel("Mean decrease in accuracy", fontdict={"size": 20})
            

        else:
            result = permutation_importance(
                MDmis_model, X_val, val_feature_table[outcome_column_name], n_repeats=10,
                random_state=21
            )
            forest_importances = pd.Series(result.importances_mean, index=X_val.columns)
            forest_importances.rename(feature_dict, inplace=True)

            forest_importances.sort_values(ascending=False, inplace=True)
            print(forest_importances.head(num_features))

            forest_importances.head(num_features).plot.barh()
            plt.xlabel("Mean decrease in accuracy", fontdict={"size": 20})
           
    else:
        if isinstance(MDmis_model, list) and len(MDmis_model) > 1:
            all_importances = []

            for model in MDmis_model:
                importances = model.feature_importances_
                all_importances.append(importances)

            importance_df = pd.DataFrame(all_importances, columns=X_val.columns)
            importance_df.rename(columns = feature_dict, inplace=True)

            avg_importances = importance_df.mean(axis=0)
            std_importances = importance_df.std(axis=0, ddof=1)
            avg_importances.sort_values(ascending=False, inplace=True)
            std_importances = std_importances.loc[avg_importances.index]
            print(avg_importances.head(num_features))

           
            avg_importances.head(num_features).plot.barh(xerr=std_importances.head(num_features), ax=ax, capsize=5, color='blue', alpha=0.75)
            plt.xlabel("Mean decrease in impurity", fontdict={"size": 20})
            

        else:
            importances = MDmis_model.feature_importances_
            forest_importances = pd.Series(importances, index=X_val.columns)
            forest_importances.rename(feature_dict, inplace=True)

            forest_importances.sort_values(ascending=False, inplace=True)
            print(forest_importances.head(num_features))

            forest_importances.head(num_features).plot.barh(color='blue', alpha=0.75)
            plt.xlabel("Mean decrease in impurity", fontdict={"size": 20})
    plt.xticks(fontsize =18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()

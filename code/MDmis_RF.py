import cProfile
import math
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

###Date: 07/08/24
###Purpose: To define modules and functions to use for model training of MDmis baseline (Random Forest).
###MDmis predicts missense variant pathogenicity
###using features derived from molecular dynamics simulations and AAindex (physiochemical features).

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)




def create_feature_table(df_table, res_data, pair_data, aa_res_matrix, aa_order, use_res_md = True, use_pair_md = True):
    aa_to_index = {aa: idx for idx, aa in enumerate(aa_order)}

    res_md_columns = [f'Res_MD_{i+1}' for i in range(47)] # number of Res MD features
    pair_md_columns = [f'Avg_Pair_MD_{i+1}' for i in range(10)] # number of Pair MD features
    aa_columns = [f'Res_AA_{i+1}' for i in range(aa_res_matrix.shape[1])]
    #column_names = md_columns + aa_columns + ['outcome']
    column_names = aa_columns

    if use_res_md:
        column_names = column_names + res_md_columns
    if use_pair_md:
        column_names = column_names + pair_md_columns

    column_names = column_names + ['outcome', 'amis_score', 'GERP_RS_score']

    feature_table = pd.DataFrame(columns=column_names)
    
  
    for index, row in df_table.iterrows():
        #query res_data
        location_variant = row["location"]
        index_protein_sequence = location_variant - int(row["start"]) 

        if row["protein_start_end"] not in res_data or row["protein_start_end"] not in pair_data:
            continue
        residue_md_features = np.array(res_data[row["protein_start_end"]][index_protein_sequence])
        
        avg_pair_md_features = np.mean(pair_data[row["protein_start_end"]][index_protein_sequence, :, :], axis = 0)


        original_aa = row["sequence"][index_protein_sequence] # string
        new_aa = row["changed_residue"] # string
        
        residue_aa_index_features = aa_res_matrix[aa_to_index[new_aa], :] - aa_res_matrix[aa_to_index[original_aa]]

        outcome = np.array(row["outcome"])
        amis_score = np.array(row["am_pathogenicity"])
        GERP_RS_score = np.array(row["GERP++_RS"])
        data_row = np.array(residue_aa_index_features)
        if use_res_md:
            data_row = np.concatenate((data_row, residue_md_features), axis=None)

        if use_pair_md:
            data_row = np.concatenate((data_row, avg_pair_md_features), axis=None)
       
        feature_table.loc[len(feature_table.index)] = np.concatenate((data_row, outcome, amis_score, GERP_RS_score),
                                                                        axis=None)
  
    return feature_table


def train_MDmis_RF(use_HGMD = False, use_res_md = True, use_pair_md = True,
                   store_model = False):
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    proteins_clinvar_mapped = pd.read_csv("/home/az2798/IDR_cons/data/amis_GERP_clinvar.csv", 
                                  index_col=0)
    #amis_clinvar = pd.read_csv("/home/az2798/IDR_cons/data/amis_clinvar.csv", 
    #                            index_col=0)

    proteins_clinvar_mapped_IDRome = proteins_clinvar_mapped[proteins_clinvar_mapped["MD Data Source"] == "IDRome"]

    if not use_HGMD:
        proteins_clinvar_mapped_IDRome = proteins_clinvar_mapped_IDRome[proteins_clinvar_mapped_IDRome["Data Source"] != "HGMD"]
    
    print("Unique protein regions before removing Xs", len(proteins_clinvar_mapped_IDRome["protein_start_end"].unique()))

    proteins_clinvar_mapped_IDRome = proteins_clinvar_mapped_IDRome[proteins_clinvar_mapped_IDRome["changed_residue"] != "X"]


    unique_proteins = proteins_clinvar_mapped_IDRome["protein_start_end"].unique()
    print("Unique protein regions", len(unique_proteins))
    unique_proteins_df = pd.DataFrame(unique_proteins, columns=["protein_start_end"])

    train_proteins, val_proteins = train_test_split(unique_proteins_df, test_size=0.1, random_state=42)

    train_data = proteins_clinvar_mapped_IDRome[proteins_clinvar_mapped_IDRome["protein_start_end"].isin(train_proteins["protein_start_end"])]
    val_data = proteins_clinvar_mapped_IDRome[proteins_clinvar_mapped_IDRome["protein_start_end"].isin(val_proteins["protein_start_end"])]

    #val_Amis = amis_clinvar[amis_clinvar["protein_start_end"].isin(val_proteins["protein_start_end"])]
    #val_Amis = val_Amis[["outcome", "am_pathogenicity"]]
    #val_Amis = val_Amis[~val_Amis["am_pathogenicity"].isna()]

    aaindex_res_mat = np.load("/home/az2798/IDR_cons/data/aa_index1.npy")

    train_feature_table = create_feature_table(train_data, res_data, pair_data, aaindex_res_mat,
                                         aa_order, use_res_md, use_pair_md)
    val_feature_table = create_feature_table(val_data, res_data, pair_data, aaindex_res_mat,
                                              aa_order, use_res_md, use_pair_md)

    if (not use_HGMD & use_res_md & use_pair_md):
        train_feature_table.to_csv("/home/az2798/IDR_cons/data/RF_train_val/train.csv")
        val_feature_table.to_csv("/home/az2798/IDR_cons/data/RF_train_val/val.csv")
        
    print(train_feature_table["outcome"].value_counts())
    print(val_feature_table["outcome"].value_counts())

    MDmis_RF = RandomForestClassifier()
    X_train = train_feature_table.drop(columns=['outcome', 'amis_score', 'GERP_RS_score'])
    y_train = train_feature_table['outcome']


    val_feature_table = val_feature_table.dropna(axis = 0, subset=["amis_score", "GERP_RS_score"])

    X_val = val_feature_table.drop(columns=['outcome', 'amis_score', 'GERP_RS_score'])
    y_val_Amis = val_feature_table[["outcome", "amis_score", 'GERP_RS_score']]

    print("X_val", X_val)
    print("y_val_Amis", y_val_Amis)

    # Train Random Forest Classifier
    MDmis_RF = RandomForestClassifier(random_state = 42)
    MDmis_RF.fit(X_train, y_train)


    ##else:
    ##    model_name = f'MDmis_RF_{model_type}_noMD'
    ##model_path = f'/home/az2798/IDR_cons/intermediate_models/{model_name}.pkl'
    if store_model:
        model_path = f'/home/az2798/IDR_cons/intermediate_models/MDmis_RF_complete.pkl'
        with open(model_path, "wb") as f:
            pickle.dump(MDmis_RF, f)

    return MDmis_RF, X_val, y_val_Amis



def calculate_model_auroc(RF_model, model_type, X_val, y_val_Amis):
    
    if model_type == "Amis":
        y_prob = y_val_Amis["amis_score"]

    elif model_type == "Amis+RF":
        y_prob = np.mean(np.array( [RF_model.predict_proba(X_val)[:, 1], y_val_Amis["amis_score"]] ), axis = 0)

    else:
        y_prob = RF_model.predict_proba(X_val)[:, 1]

    auroc = roc_auc_score(y_val_Amis["outcome"], y_prob)
    fpr, tpr, _ = roc_curve(y_val_Amis["outcome"], y_prob)

    return fpr, tpr, auroc



def plot_rocs(fpr_vals, tpr_vals, auroc_vals, labels, file_path):
    plt.figure()
    for i in range(len(fpr_vals)):
        plt.plot(fpr_vals[i], tpr_vals[i], label=f'{labels[i]}: AUROC = {auroc_vals[i]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(file_path, dpi = 300, bbox_inches = 'tight')

def plot_feature_importance(MDmis_RF, X_val, y_val_Amis):
    """
    Plot the top 10 feature importances of a trained RandomForest model.

    Parameters:
    -----------
    MDmis_RF : RandomForestClassifier
        The trained RandomForest model.
    
    X_val : DataFrame
        The validation set features.
    
    y_val_Amis : DataFrame
        The validation set target, including 'outcome', 'amis_score', and GERP_RS_score.

    Returns:
    --------
    None
        This function saves a plot of the top 10 feature importances to a file.
    """
    #Calculate feature importances
    feature_importance = MDmis_RF.feature_importances_
    std_dev = np.std([tree.feature_importances_ for tree in MDmis_RF.estimators_], axis=0)
    
    #Get the top 10 features
    sorted_idx_features = np.argsort(feature_importance)[-10:]
    top_features = X_val.columns[sorted_idx_features]

    #Plot the top 10 features
    plt.figure(figsize=(10, 6))
    plt.barh(range(10), feature_importance[sorted_idx_features], xerr=std_dev[sorted_idx_features], align='center')
    plt.yticks(range(10), top_features)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Top 10 Feature Importances")
    plt.savefig("/home/az2798/IDR_cons/results/top_10_feature_importances.png", dpi=300, bbox_inches='tight')
    

def compare_MDmis_AM(MDmis_RF, X_val, y_val_Amis, file_path):
    """
    Plot a correlation plot for low and high constraint showing MDmis-RF scores and AM scores
    Parameters:
    -----------
    MDmis_RF : RandomForestClassifier
        The trained RandomForest model.
    
    X_val : DataFrame
        The validation set features.
    
    y_val_Amis : DataFrame
        The validation set target, including 'outcome', 'amis_score', and GERP_RS_score.

    file_path: str
        The string to use to save the resulting figure
    Returns:
    --------
    None
        This function saves a plot of the top 10 feature importances to a file.
    """
    y_prob = MDmis_RF.predict_proba(X_val)[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_prob, y = y_val_Amis["amis_score"])
    plt.xlabel("MDmis-RF Score")
    plt.ylabel("AlphaMissense Score")
    plt.grid(True)

    plt.savefig(file_path, dpi =300,
                bbox_inches = "tight")    

def train_model_store_roc(use_HGMD, model_type, use_res_md, use_pair_md, GERP_cutoff= None):
    store_model = False
    if model_type == "RF" and use_res_md and use_pair_md:
        store_model = True
    MDmis_RF, X_val, y_val_Amis = train_MDmis_RF(use_HGMD, use_res_md, use_pair_md, store_model)

    if GERP_cutoff == None:
        fpr, tpr, auroc = calculate_model_auroc(MDmis_RF, model_type, X_val, y_val_Amis)
        fpr_vals.append(fpr)
        tpr_vals.append(tpr)
        auroc_vals.append(auroc)
        if model_type == "RF" and use_res_md and use_pair_md:
            plot_feature_importance(MDmis_RF, X_val,y_val_Amis)
    
    else:
        low_conserved_indices = y_val_Amis[y_val_Amis["GERP_RS_score"] < GERP_cutoff].index
        high_conserved_indices = y_val_Amis[y_val_Amis["GERP_RS_score"] >= GERP_cutoff].index

        low_conserved_X_val = X_val.loc[low_conserved_indices]
        low_conserved_y_val = y_val_Amis.loc[low_conserved_indices]

        high_conserved_X_val = X_val.loc[high_conserved_indices]
        high_conserved_y_val = y_val_Amis.loc[high_conserved_indices]

        fpr, tpr, auroc = calculate_model_auroc(MDmis_RF, model_type, low_conserved_X_val, 
                                                low_conserved_y_val)
        fpr_vals_low_cons.append(fpr)
        tpr_vals_low_cons.append(tpr)
        auroc_vals_low_cons.append(auroc)

        fpr, tpr, auroc = calculate_model_auroc(MDmis_RF, model_type, high_conserved_X_val, 
                                                high_conserved_y_val)
        fpr_vals_high_cons.append(fpr)
        tpr_vals_high_cons.append(tpr)
        auroc_vals_high_cons.append(auroc)

        if model_type == "RF" and use_res_md and use_pair_md:
            compare_MDmis_AM(MDmis_RF, low_conserved_X_val, low_conserved_y_val,
                             "/home/az2798/IDR_cons/results/low_cons_comparison_scores.png")
            compare_MDmis_AM(MDmis_RF, high_conserved_X_val, high_conserved_y_val,
                             "/home/az2798/IDR_cons/results/high_cons_comparison_scores.png")

    
def load_h5py_data(h5py_file):
    """
    Loads residue and pair data from h5py file.

    Returns:
        dict: Dictionary containing residue data.
        dict: Dictionary containing pair data.
    """
    res_data = {}
    pair_data = {}

    with h5py.File(h5py_file, 'r') as f:
        for key in f.keys():
            if key.endswith('_res'):
                protein_start_end = key.replace("_res", "")
                res_data[protein_start_end] = np.array(f[key][:])
            elif key.endswith('_pair'):
                protein_start_end = key.replace("_pair", "")
                pair_data[protein_start_end] = np.array(f[key][:])

    return res_data, pair_data
    

def main():
    global res_data, pair_data

    global fpr_vals, tpr_vals, auroc_vals

    global fpr_vals_low_cons, tpr_vals_low_cons, auroc_vals_low_cons

    global fpr_vals_high_cons, tpr_vals_high_cons, auroc_vals_high_cons

    h5py_path = "/share/vault/Users/az2798/train_data_all/filtered_feature_all_ATLAS_GPCRmd_IDRome.h5"
    rocs_path = "/home/az2798/IDR_cons/results/MDmis_RF_rocs/"
    
    res_data, pair_data = load_h5py_data(h5py_path)

    fpr_vals, tpr_vals, auroc_vals = [], [], []

    fpr_vals_low_cons, tpr_vals_low_cons, auroc_vals_low_cons = [], [], []

    fpr_vals_high_cons, tpr_vals_high_cons, auroc_vals_high_cons = [], [], []

    GERP_cutoff = None
    train_model_store_roc(use_HGMD = False, model_type = "RF", use_res_md=True, use_pair_md=True, GERP_cutoff=GERP_cutoff)
    train_model_store_roc(use_HGMD = False, model_type = "RF", use_res_md=True, use_pair_md=False, GERP_cutoff=GERP_cutoff)

    train_model_store_roc(use_HGMD = False, model_type = "RF", use_res_md=False, use_pair_md=False, GERP_cutoff=GERP_cutoff)
    train_model_store_roc(use_HGMD = False, model_type = "Amis", use_res_md=False, use_pair_md=False, GERP_cutoff=GERP_cutoff)

    train_model_store_roc(use_HGMD = False, model_type = "Amis+RF", use_res_md=True, use_pair_md=True, 
                          GERP_cutoff=GERP_cutoff)

    labels = ["MDmis (Res + Pair + AAIndex)", "MDmis (Res + AAIndex)",
              "MDmis (AAIndex)", "AlphaMissense", "AlphaMissense + MDmis"]
    
    #calculate_auroc(MDmis_RF_model, X_val, y_val)

    #plot_rocs(fpr_vals, tpr_vals, auroc_vals, labels, f'{rocs_path}all_test.png')

    plot_rocs(fpr_vals_low_cons, tpr_vals_low_cons, auroc_vals_low_cons, labels, f'{rocs_path}low_cons.png')
    
    plot_rocs(fpr_vals_high_cons, tpr_vals_high_cons, auroc_vals_high_cons, labels, f'{rocs_path}high_cons.png')



    #test_linear()

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.inspection import permutation_importance
import h5py
import scipy.stats as ss
from ridgeplot import ridgeplot
from itertools import combinations
from scipy.stats import mannwhitneyu, sem

def create_feature_table(df_table, res_data, pair_data, aa_res_matrix,
                        aa_order, window_size = 3,
                        location_column_name = None, 
                        label_source_column_name = None,
                        original_aa_column_name = None,
                        changed_aa_column_name = None,
                        outcome_column_name = None):
    """
    Creates a feature table combining residue-level and pairwise molecular dynamics (MD) data,
    amino acid (AA) features, and additional variant-specific information for each protein variant.

    Parameters:
    ----------
    df_table : pandas.DataFrame
        A DataFrame containing information about protein variants, including location, sequence, 
        and metadata like "outcome", "location", etc.
        
    res_data : dict
        A dictionary mapping protein regions ("protein_start_end") to corresponding residue-level MD features.
        
    pair_data : dict
        A dictionary mapping protein regions ("protein_start_end") to corresponding pairwise MD features.
    

    aa_res_matrix : numpy.ndarray
        A matrix that represents amino acid-specific features, where rows correspond to amino acids, 
        and columns correspond to their respective feature vectors.
        
    aa_order : list
        A list of amino acids in the same order as the indices of `aa_res_matrix`, used to map amino acids 
        to feature vectors.
                
    location_column_name : str
        The column name that should be used to access the variant's location in the sequence

    label_source_column_name : str
        The column name that should be used to access the variant's label source

    original_aa_column_name : str
        The column name that should be used to access the original AA in the sequence
    
    changed_aa_column_name : str
        The column name that should be used to access the changed AA in the sequence
    
    outcome_column_name : str
        The column name that should be used to access the outcome label 

    Returns:
    --------
    feature_table : pandas.DataFrame
        A DataFrame containing the extracted features for each variant, including amino acid features, 
        residue-level MD features, pairwise MD features (if applicable), and additional variant-specific information.

    Notes:
    ------
    - Residue-level and pairwise MD features are extracted based on the location of the variant in the sequence.
    - If MD data for a protein is missing in either `res_data` or `pair_data`, the variant is skipped.
    - The final feature table combines AA features, MD features (if applicable), and variant-specific metadata.
    """

    aa_to_index = {aa: idx for idx, aa in enumerate(aa_order)}
    aa_to_onehot = {aa: np.eye(len(aa_order))[i] for i, aa in enumerate(aa_order)}

    # aa_columns = [f'Res_AA_{i+1}' for i in range(aa_res_matrix.shape[1])]
    # res_md_columns = [f'Res_MD_{i+1}' for i in range(47)] # number of Res MD features
    # pair_md_columns = [f'Pair_MD_{i+1}' for i in range(10)] # number of Pair MD features

    
    rows = []

    for index, row in df_table.iterrows():
        #query res_data
        location_variant = row[location_column_name]
        index_protein_sequence = location_variant - int(row["start"]) 

        if row["protein_start_end"] not in res_data or row["protein_start_end"] not in pair_data:
            continue
        
        residue_array = res_data[row["protein_start_end"]].copy()
        average_RMSF = np.mean(residue_array[:, 2])
        average_SASA = np.mean(residue_array[:, 0])
        residue_array[:, 2] /= average_RMSF  #Normalize RMSF



        residue_md_features, residue_md_labels = get_window_with_padding(
        residue_array,
        index_protein_sequence,
        window_size
        )
        #avg_pair_md_features = np.mean(pair_data[row["protein_start_end"]][index_protein_sequence, :, :], axis = 0)
        ### Pair features binarization, only for interactions
        interaction_cutoff = 0.4
        pair_data_above_cutoff = pair_data[row["protein_start_end"]][:,:,:9] > interaction_cutoff
        # count_pair_md_features, pair_labels = get_pair_window_with_padding_and_labels(pair_data_above_cutoff, index_protein_sequence,
        #                                         window_size)
        count_pair_md_features = (
        np.sum(pair_data_above_cutoff[index_protein_sequence, :, :9], axis=(0)) + 
        np.sum(pair_data_above_cutoff[:, index_protein_sequence, :9], axis=(0))   
        )

        z_scored_cov = ss.zscore(pair_data[row["protein_start_end"]][: , :, -1], axis=None)
        avg_cov = (np.mean(z_scored_cov[index_protein_sequence,:]) + np.mean(z_scored_cov[:, index_protein_sequence])) /2
        print(avg_cov, row[outcome_column_name])
        
        if original_aa_column_name is None:
            original_aa = row["sequence"][index_protein_sequence] # string
        else:
            original_aa = row[original_aa_column_name]
        new_aa = row[changed_aa_column_name] # string
        
        
        #print(res_data[row["protein_start_end"]][:, 0].shape, f"Shape of {row["protein_start_end"]}")
        

        residue_aa_index_features = aa_res_matrix[aa_to_index[new_aa], :] - aa_res_matrix[aa_to_index[original_aa]]

        one_hot_diff = aa_to_onehot[new_aa] - aa_to_onehot[original_aa]

        if label_source_column_name is None:
            label_source = np.nan
        else:
            label_source = row[label_source_column_name]
        rows.append({
                    **{f"Res_AA_{j+1}": val for j, val in enumerate(residue_aa_index_features)},
                    **{label: val for label, val in zip(residue_md_labels, residue_md_features)},
                    **{f"Pair_MD_{j+1}": val for j, val in enumerate(count_pair_md_features)},
                    "Pair_MD_10": avg_cov,
                    **{f"One_Hot_{j+1}": val for j, val in enumerate(one_hot_diff)},
                    "outcome": row[outcome_column_name],
                    "UniProtID": row["UniProtID"],
                    "protein_start_end": row["protein_start_end"],
                    "location": location_variant,
                    "Label Source": label_source,
                    "Original AA": original_aa,
                    "Changed AA": new_aa,
                    "Average SASA": average_SASA,
                    "Average RMSF": average_RMSF
                })
    # Build DataFrame at once
    feature_table = pd.DataFrame(rows)
    return feature_table

def get_window_with_padding(data_array, center_idx, window_size):
    start = center_idx - window_size
    end = center_idx + window_size + 1 
    padded = np.zeros((2 * window_size + 1, data_array.shape[1]))

    valid_start = max(start, 0)
    valid_end = min(end, data_array.shape[0])

    insert_start = valid_start - start 
    insert_end = insert_start + (valid_end - valid_start)

    padded[insert_start:insert_end] = data_array[valid_start:valid_end]
    labels = []
    for i in range(2 * window_size + 1):
        offset = i - window_size
        for j in range(data_array.shape[1]):
            labels.append(f"Res_MD_{j+1}_pos_{offset}")

    return padded.flatten(), labels


def load_MD_data(h5py_file):
    """
    Loads residue and pair MD data from h5py file.

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


def split_rows(df, columns_to_split, char_to_split):
    #Split the specified columns by ';' and create a new dataframe by exploding the lists
    split_df = df.assign(**{col: df[col].str.split(char_to_split) for col in columns_to_split})
    exploded_df = split_df.explode(columns_to_split, ignore_index=True)
    return exploded_df


def plot_rocs(val_feature_table, list_of_y_probs, outcome_column_name, title,
              labels, figure_path):
    """
    Plots the ROC curves for multiple sets of predicted probabilities, calculating and displaying AUROC values.

    Parameters:
    ----------
    val_feature_table : pandas.DataFrame
        A DataFrame containing the true outcomes for the validation set.
        
    list_of_y_probs : list of numpy.ndarray
        A list of predicted probability arrays, where each array corresponds to a different model or set of predictions.
        
    outcome_column_name : str
        The name of the column in `val_feature_table` that contains the true binary outcomes (e.g., "outcome").
    
    title : str
        The title for the ROC curves to describe the validation set used

    labels : list of str
        A list of labels corresponding to each set of predicted probabilities, used to annotate the ROC curves.
        
    figure_path : str
        The file path where the resulting ROC plot should be saved.

    Returns:
    --------
    None
        Saves the ROC plot as an image file at the specified path.
    
    Notes:
    ------
    - The ROC curve is plotted for each set of predicted probabilities, and the AUROC score is displayed in the legend.
    """

    
    plt.figure()
    for i, y_prob in enumerate(list_of_y_probs):
        auroc = roc_auc_score(val_feature_table[outcome_column_name], y_prob)
        fpr, tpr, _ = roc_curve(val_feature_table[outcome_column_name], y_prob)
        plt.plot(fpr, tpr, label=f'{labels[i]}: AUROC = {auroc:.2f}')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title, fontdict={"size": 14})
    plt.legend(loc='lower right')
    plt.savefig(figure_path, dpi = 300, bbox_inches = 'tight')    


def plot_average_roc(feature_table_with_predictions,
                     predictions_labels_dict,
                     num_folds,
                     fold_column_name,
                     outcome_column_name,
                     FPR_cutoff, title, figure_path):
    """
    Plots the average ROC curve across folds with standard error shading, calculating and displaying AUROC values.

    Parameters:
    ----------
    feature_table_with_predictions : pandas.DataFrame
        A pandas DataFrame containing the true outcomes for the validation sets for each fold.
        
    predictions_labels_dict : dict
        A list of labels corresponding to each set of predicted probabilities, used to annotate the ROC curves.

    outcome_column_name : str
    
        The name of the column in the validation DataFrame that contains the true binary outcomes (e.g., "outcome").
    FPR_cutoff: float
        The value to use to compute AUROC. If considering all FPR, then this should be 1, but for high precision, use 0.05/0.1

    title : str
        The title for the ROC curves to describe the validation set used.

    figure_path : str
        The file path where the resulting ROC plot should be saved.

    Returns:
    --------
    None
        Saves the ROC plot as an image file at the specified path.
    """
    print(feature_table_with_predictions[outcome_column_name].value_counts())
    plt.figure(figsize=(10, 8))
    
    for prediction_column in predictions_labels_dict.keys():  #Loop over models
        all_fpr = np.linspace(0, 1, 100)  #Define fixed FPR points for interpolation
        tprs = []  #Store TPRs for each fold
        aucs = []  #Store AUROC for each fold

        for fold in range(1, num_folds + 1):
            fold_subset = feature_table_with_predictions[feature_table_with_predictions[fold_column_name] == fold]
            fpr, tpr, _ = roc_curve(fold_subset[outcome_column_name], fold_subset[prediction_column])
            tpr_interp = np.interp(all_fpr, fpr, tpr)  #Interpolate TPRs at fixed FPR points
            tpr_interp[0] = 0.0  #Ensure TPR starts at 0
            tprs.append(tpr_interp)
            
            if FPR_cutoff is None:
                aucs.append(roc_auc_score(fold_subset[outcome_column_name], fold_subset[prediction_column]))
            else: 
                mask = all_fpr <= FPR_cutoff
                fpr_filtered = all_fpr[mask]
                tpr_filtered = tpr_interp[mask]

                if len(fpr_filtered) > 1:  #Ensure there are enough points to compute partial AUC
                    partial_auc = np.trapz(tpr_filtered, fpr_filtered, dx = 0.001)  
                    normalized_auc = partial_auc * (1 / FPR_cutoff)  
                else:
                    normalized_auc = np.nan  

                aucs.append(normalized_auc)

        mean_tpr = np.mean(tprs, axis=0)  
        sem_tpr = np.std(tprs, axis= 0, ddof=1) / np.sqrt(np.size(tprs))

        mean_auc = np.mean(aucs)       
        sem_auc = np.std(aucs, axis=0, ddof = 1) / np.sqrt(np.size(aucs))          

        #Plot mean ROC curve
        plt.plot(all_fpr, mean_tpr, label=f'{predictions_labels_dict[prediction_column]}: {mean_auc:.2f}±{sem_auc:.3f}')
        #Plot standard error shading
        plt.fill_between(all_fpr, mean_tpr - sem_tpr, mean_tpr + sem_tpr, alpha=0.2)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.xlabel('False Positive Rate', fontdict={"size":18})
    plt.ylabel('True Positive Rate', fontdict={"size":18})
    plt.title(title, fontdict={"size": 18})
    if FPR_cutoff is None:
        plt.legend(title = "AUROC",loc='lower right',title_fontsize='18', fontsize=16)
    else:
        plt.legend(title = f"AUROC({FPR_cutoff})",loc='lower right', title_fontsize = '18',fontsize=16)

    plt.savefig(figure_path, dpi=300, bbox_inches='tight')




def plot_rhos_by_Uniprot(val_feature_table, list_of_y_probs, outcome_column_name,
              UniProtID_column_name, title, labels, figure_path):
    """
    Plots violin plots for multiple sets of predicted probabilities, calculating Spearman rho
    for each UniProtID and displaying the distributions.

    Parameters:
    ----------
    val_feature_table : pandas.DataFrame
        A DataFrame containing the true outcomes and UniProtIDs for the validation set.
        
    list_of_y_probs : list of numpy.ndarray
        A list of predicted probability arrays, where each array corresponds to a different model or set of predictions.
        
    outcome_column_name : str
        The name of the column in `val_feature_table` that contains the true continuous outcomes (e.g., "outcome").

    UniProtID_column_name : str
        The name of the column in `val_feature_table` that contains the UniProt IDs.
    
    title : str
        The title for the violin plots to describe the validation set used.

    labels : list of str
        A list of labels corresponding to each set of predicted probabilities, used to annotate each violin.

    figure_path : str
        The file path where the resulting plot should be saved.

    Returns:
    --------
    None
        Saves the plot as an image file at the specified path.
    """
    
    rho_data = []
    for i, y_prob in enumerate(list_of_y_probs):
        #Calculate Spearman rho for each UniProtID
        for uniprot_id in val_feature_table[UniProtID_column_name].unique():
            subset = val_feature_table[val_feature_table[UniProtID_column_name] == uniprot_id]
            rho, _ = ss.spearmanr(subset[outcome_column_name], y_prob[subset.index])
            rho_data.append({'Model': labels[i], "UniProtID": uniprot_id, 'Spearman Rho': rho})

    rho_df = pd.DataFrame(rho_data)
    print(rho_data)
    plt.figure(figsize=(12, 7))
    sns.violinplot(x="Model", y="Spearman Rho", data=rho_df, inner="point", palette="Set2")
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)

    for uniprot_id in rho_df["UniProtID"].unique():
        subset = rho_df[rho_df["UniProtID"] == uniprot_id]
        if len(subset) > 1:  
            plt.plot(
                subset["Model"],
                subset["Spearman Rho"],
                color='gray',
                linewidth=0.8,
                alpha=0.6,
                marker="o",
            )

    plt.title(title, fontdict={"size": 14})
    plt.ylabel("Spearman Rho")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(figure_path, dpi=300, bbox_inches='tight')

    plt.clf()

    fig, axes = plt.subplots(2, 3, figsize=(8,12))

    axes = axes.flatten()

    for i, y_prob in enumerate(list_of_y_probs):
        ax = axes[i] 
        sns.scatterplot(
            x=y_prob[val_feature_table.index],
            y=val_feature_table[outcome_column_name],
            ax=ax,
            alpha=0.6,
        )
        ax.set_title(f"Model: {labels[i]}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("True Outcome")
        ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8)  


    fig.savefig(figure_path.replace(".png", "_scatter_grid.png"), dpi=300, bbox_inches="tight")

def plot_rhos_by_group(val_feature_table, list_of_y_probs, outcome_column_name,
              group_column_name, title, labels, figure_path):
    """
    Plots violin plots for multiple sets of predicted probabilities, calculating Spearman rho
    for each UniProtID and displaying the distributions.

    Parameters:
    ----------
    val_feature_table : pandas.DataFrame
        A DataFrame containing the true outcomes and UniProtIDs for the validation set.
        
    list_of_y_probs : list of numpy.ndarray
        A list of predicted probability arrays, where each array corresponds to a different model or set of predictions.
        
    outcome_column_name : str
        The name of the column in `val_feature_table` that contains the true continuous outcomes (e.g., "outcome").

    group_column_name : str
        The name of the column in `val_feature_table` that contains the UniProt IDs.
    
    title : str
        The title for the violin plots to describe the validation set used.

    labels : list of str
        A list of labels corresponding to each set of predicted probabilities, used to annotate each violin.

    figure_path : str
        The file path where the resulting plot should be saved.

    Returns:
    --------
    None
        Saves the plot as an image file at the specified path.
    """
    
    rho_data = []
    for i, y_prob in enumerate(list_of_y_probs):
        #Calculate Spearman rho for each UniProtID
        for group in val_feature_table[group_column_name].unique():
            subset = val_feature_table[val_feature_table[group_column_name] == group]
            rho, p_val = ss.spearmanr(subset[outcome_column_name], y_prob[subset.index])
            rho_data.append({'Model': labels[i], "Group": group, 'Spearman Rho': rho,
                             'p-value': p_val, 'N': len(subset)})

    rho_df = pd.DataFrame(rho_data)
    print(rho_df)
    plt.figure(figsize=(12, 7))
    sns.violinplot(x="Model", y="Spearman Rho", data=rho_df, inner="point", palette="Set2")
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)

    for group in rho_df["Group"].unique():
        subset = rho_df[rho_df["Group"] == group]
        if len(subset) > 1:  
            plt.plot(
                subset["Model"],
                subset["Spearman Rho"],
                color='gray',
                linewidth=0.8,
                alpha=0.6,
                marker="o",
            )

    plt.title(title, fontdict={"size": 14})
    plt.ylabel("Spearman Rho")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(figure_path, dpi=300, bbox_inches='tight')



def pairwise_mannwhitneyu(data, group_col, value_col, verbose = False):
    """
    Perform pairwise MannWhitney U tests for all unique pairs in a given column.
    
    Parameters:
    - data: pandas DataFrame with the data.
    - group_col: str, name of the column with categorical data.
    - value_col: str, name of the column with numerical data to test.
    - verbose; boolean, whether to print p-values
    
    Returns:
    - results: Dictionary with pairs as keys and p-values as values.
    """

    groups = data[group_col].unique()
    p_vals, stats = {}, {}
    num_tests = len(groups) * (len(groups) - 1) / 2
    #Loop over all pairs of groups
    for (g1, g2) in combinations(groups, 2):
        data1 = data[data[group_col] == g1][value_col]
        data2 = data[data[group_col] == g2][value_col]
        
        stat, p_value = mannwhitneyu(data1, data2, nan_policy = "omit")
        adjusted_p = p_value * num_tests
        p_vals[(g1, g2)] = min(adjusted_p, 1.0)
        stats[(g1,g2)] = stat
    if verbose:
        print(p_vals)
        print(stats)
    return p_vals, stats


def get_significance_stars(p_value, significance_levels):
    for threshold, stars in significance_levels.items():
        if p_value <= threshold:
            return stars
    return ""  

def plot_ridgeplot(data, group_col, value_col, results_dir, plot_filename,
                   significance_levels={0.0001: '***', 0.001: '**', 0.05: '*'},
    xlabel='Average Cosine Similarity of Composition',
    label_placement = "left", xlim = None, 
    bar_height=0.2, xlim_buffer=1.3, verbose = False, type = "kde", palette=None):
    
    p_vals, stats = pairwise_mannwhitneyu(data, group_col, value_col)
    groups = data[group_col].unique()
    groups.sort()  
    num_groups = len(groups)

    fig, ax = plt.subplots(nrows = 3, figsize=(8, 6), sharex = True) 

    if palette is None:
        palette = sns.color_palette("husl", num_groups)  


    for i, group in enumerate(groups):
        subset = data[data[group_col] == group]
        if palette is None:
            color = palette[i]
        else:
            color = palette[group]

        if type =="kde":
            sns.kdeplot(subset[value_col], ax=ax[i], fill=True, alpha=0.9, 
                        linewidth=1, color=color, bw_adjust=1.5
                        )
            
            kde = sns.kdeplot(subset[value_col], ax=ax[i], color="w", lw=2, bw_adjust=1.5
                        )
            x_values, y_values = kde.get_lines()[0].get_data()
            mean_x = subset[value_col].mean()
            median_x = np.median(subset[value_col])

            mean_y = np.interp(mean_x, x_values, y_values)*0.99
            median_y = np.interp(median_x, x_values, y_values)*0.99

            ax[i].plot([mean_x, mean_x], [0, mean_y], c="k", ls="-", lw=2.5)
            ax[i].plot([median_x, median_x], [0, median_y], c="k", ls="--", lw=2.5)

        else:
            unique_values = np.sort(subset[value_col].unique())
            value_counts = subset[value_col].value_counts().sort_index()

            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax[i], color=color)
            bar_positions = np.arange(0, len(value_counts.index ))
            ax[i].set_xticks(bar_positions)
        group_label = group.replace(" - ", "\n")

        if label_placement == "left":
        
            ax[i].text(0, 
                0.2, 
                group_label, ha="left", va="center", fontweight="bold",
                  fontsize=18, color="black", transform = ax[i].transAxes)
        else:
            ax[i].text(1, 
                0.2, 
                group_label, ha="right", va="center", fontweight="bold",
                  fontsize=18, color="black", transform = ax[i].transAxes)
        
        ax[i].set_yticks([])
        ax[i].set_yticklabels([])
        
        ax[i].set_xlabel("")
        ax[i].set_ylabel("")

        sns.despine(left=True)

    fig.supxlabel(xlabel, fontweight = "bold", fontsize=18)
    if type == "kde":
        ax[-1].tick_params(axis="x", which="both", labelsize=18)
    else:
        
        ax[-1].set_xticklabels(unique_values.astype(int), ha="center")
    if xlim:
        plt.xlim(xlim)

    
    plt.subplots_adjust(hspace = -1)
    plt.tight_layout()

    xmax = xlim_buffer
    for (g1, g2), p in p_vals.items():
        stars = get_significance_stars(p, significance_levels)
        if stars:  
            x1, x2 = groups.tolist().index(g1), groups.tolist().index(g2)
            con = ConnectionPatch(xyA=(xmax, 0.5), coordsA=ax[x1].transAxes,
                      xyB=(xmax,0.5), coordsB=ax[x2].transAxes, linewidth=2)
            fig.add_artist(con)
            y1 = ax[x1].get_position().y0 + ax[x1].get_position().height / 2
            y2 = ax[x2].get_position().y0 + ax[x2].get_position().height / 2
            
            x_max_fig = fig.transFigure.inverted().transform(
                (ax[x1].transAxes.transform((xmax, 0))))[0]
            fig.text(x_max_fig, (y1 + y2)/2, stars, ha='center', va='center', rotation="vertical", 
                    fontsize=20, color="black", transform=fig.transFigure)
            xmax = xmax+bar_height
    plt.savefig(os.path.join(results_dir, plot_filename), dpi=300, bbox_inches="tight", pad_inches = 0.2)
    plt.close()

def plot_boxplot_with_significance(
    data, group_col, value_col, results_dir, plot_filename,
    significance_levels={0.0001: '***', 0.001: '**', 0.05: '*'},
    title=None, xlabel='RMSF Category', ylabel='Average Cosine Similarity of Composition',
    bar_height=0.2, ylim_buffer=1.3, rotation=45, verbose = False, palette=None,
    plot_type="box"
):
    """
    Plot a boxplot with pairwise significance testing and annotate with significance stars.
    """

    #Perform pairwise tests
    p_vals, stats = pairwise_mannwhitneyu(data, group_col, value_col,
                                          verbose)

    data.sort_values(by = group_col, inplace = True)
    #Select plot type
    if plot_type == "box":
        if palette is not None:
            ax =sns.boxplot(data=data, x=group_col, y=value_col, palette=palette)
        else:
            ax=sns.boxplot(data=data, x=group_col, y=value_col)
    else:
        if palette is not None:
            ax=sns.violinplot(data=data, x=group_col, y=value_col, palette=palette)
        else:
            ax= sns.violinplot(data=data, x=group_col, y=value_col)
    if title is not None:
        plt.title(title)
    

    num_significant_pairs = sum(1 for p in p_vals.values() if get_significance_stars(p, significance_levels))
    additional_padding = num_significant_pairs * bar_height
    
    plt.xlabel(xlabel, fontsize = 18, fontweight = "bold")
    plt.ylabel(ylabel, fontsize = 18, fontweight = "bold")
    plt.xticks(rotation=rotation, ha="right")
    ax.set_xticklabels(labels = [xtl.get_text().replace(" - ", "\n") for xtl in ax.get_xticklabels()])
    ax.tick_params(axis='both', labelsize=18)

    plt.grid()

    #Determine y-limits and compute additional padding
    y_max = data[value_col].max()
    y_min = data[value_col].min()

    #Estimate the number of significance bars to adjust y-limit
    

    plt.ylim(y_min, y_max * ylim_buffer + additional_padding)

    y = y_max * ylim_buffer
    for (g1, g2), p in p_vals.items():
        stars = get_significance_stars(p, significance_levels)
        if stars:  # Only add bar if significant
            x1, x2 = data[group_col].unique().tolist().index(g1), data[group_col].unique().tolist().index(g2)
            plt.plot([x1, x2], [y, y], color="black", linewidth =2)
            plt.text((x1 + x2) * 0.5, y, stars, ha='center', va='bottom', color="black", fontsize=16)
            y += bar_height  # Increment for the next bar
    plt.tight_layout() 
     
    plt.savefig(os.path.join(results_dir, plot_filename), dpi=300, bbox_inches="tight")
    plt.close()





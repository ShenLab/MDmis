import sys

sys.path.append('/home/az2798/MDmis/code/') #change to your path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import matplotlib
import pandas as pd
from matplotlib.colors import ListedColormap
import scipy.stats as ss
import numpy as np
import os
import re
import warnings
from sklearn.linear_model import LinearRegression
from utils import *



warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.rcParams.update({'font.size': 13})
pd.set_option('display.max_columns', None)




def plot_GERP_info(IDRs_table, other_regions_table,
                   GERP_column_name, alphamissense_score_column,
                   outcome_column_name, variant_palette,
                   results_dir):
    matplotlib.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (14,6))
    
    sns.histplot(IDRs_table[GERP_column_name], bins=30, kde=False, ax=ax[0])
    ax[0].set_xlabel("GERP++_RS")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title("Histogram of GERP++_RS (IDRs)")
    ax[0].grid(True)

    sns.histplot(other_regions_table["GERP++_RS"], bins=30, kde=False, ax = ax[1])
    ax[1].set_xlabel("GERP++_RS")
    ax[1].set_ylabel("Frequency")
    ax[1].set_title("Histogram of GERP++_RS (Other Protein Regions)")
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(
        results_dir,
        "GERP_RS_hist_protein_types.png", 
        ), dpi=300, bbox_inches="tight")
    plt.clf()
    ##########
    benign_IDRs = IDRs_table[
        IDRs_table["Variant Effect"] == "Benign"
    ][GERP_column_name].dropna()

    pathogenic_MD = IDRs_table[
        IDRs_table["Variant Effect"] == "Pathogenic"
    ][GERP_column_name].dropna()

    u_stat_MD, p_value_MD = ss.mannwhitneyu(benign_IDRs, pathogenic_MD, alternative='two-sided')
    print(u_stat_MD, p_value_MD, "MD table Test for GERP")

    plt.figure(figsize=(8,6))
    ax =plt.gca()
    sns.histplot(
    data=IDRs_table,
    x="GERP++_RS",
    hue= outcome_column_name,
    bins=30,
    kde=False,
    stat="density",
    common_norm=False,    
    palette=variant_palette,
    alpha=0.6,
    )
    ax.tick_params(labelsize = 20)
    ax.xaxis.label.set_size(22)
    ax.yaxis.label.set_size(22)
    plt.axvline(x=2, color='r', linestyle='--')
    plt.setp(ax.get_legend().get_texts(), fontsize='22') 
    plt.setp(ax.get_legend().get_title(), fontsize='22') 
    plt.tight_layout()
    plt.savefig(os.path.join(
        results_dir,
        "GERP_RS_dist_variant_type.png"), dpi=300, bbox_inches="tight")
    plt.clf()
    ##########


    benign_others = other_regions_table[
        other_regions_table["Variant Effect"] == "Benign"
    ][GERP_column_name].dropna()

    pathogenic_others = other_regions_table[
        other_regions_table["Variant Effect"] == "Pathogenic"
    ][GERP_column_name].dropna()

    u_stat_others, p_value_others = ss.mannwhitneyu(benign_others, pathogenic_others, alternative='two-sided')
    print(u_stat_others, p_value_others, "Others table Test for GERP")

    plt.figure(figsize=(8,6))
    ax =plt.gca()
    sns.histplot(
    data=other_regions_table,
    x="GERP++_RS",
    hue= outcome_column_name,
    bins=30,
    kde=False,
    stat="density",
    common_norm=False,    
    palette=variant_palette,
    alpha=0.6,
    )
    ax.tick_params(labelsize = 20)
    ax.xaxis.label.set_size(22)
    ax.yaxis.label.set_size(22)
    plt.axvline(x=2, color='r', linestyle='--')
    plt.setp(ax.get_legend().get_texts(), fontsize='22') 
    plt.setp(ax.get_legend().get_title(), fontsize='22') 
    plt.tight_layout()
    plt.savefig(os.path.join(
        results_dir,
        "GERP_RS_dist_variant_type_other_prots.png"), dpi=300, bbox_inches="tight")
    plt.clf()

    #####
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (12,6), sharey=True)
    sns.kdeplot(
        data = IDRs_table,
        x="GERP++_RS", y=alphamissense_score_column, ax=ax[0])
    ax[0].set_xlabel("GERP++_RS", fontsize = 18)
    ax[0].set_ylabel("AlphaMissense Score",fontsize=18)
    ax[0].set_title("IDRs", fontsize=18)
    ax[0].grid(True)

    sns.kdeplot(
        data = other_regions_table,
        x="GERP++_RS", y=alphamissense_score_column, ax = ax[1])
    ax[1].set_xlabel("GERP++_RS", fontsize=18)
    ax[1].set_ylabel("AlphaMissense Score", fontsize=18)
    ax[1].set_title("Other Protein Regions", fontsize=18)
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(
        results_dir,
        "GERP_RS_AM_corr.png"), dpi =300,
                bbox_inches = "tight")


def plot_pLDDT_info(MD_table, other_regions_table,
                    title_for_MD_table,
                    variant_palette,
                   GERP_column_name, 
                   pLDDT_column_name,
                   results_dir):
    matplotlib.rcParams.update({'font.size': 18})
    benign_MD = MD_table[
        MD_table["Variant Effect"] == "Benign"
    ][pLDDT_column_name].dropna()

    pathogenic_MD = MD_table[
        MD_table["Variant Effect"] == "Pathogenic"
    ][pLDDT_column_name].dropna()

    u_stat_MD, p_value_MD = ss.mannwhitneyu(benign_MD, pathogenic_MD, alternative='two-sided')
    print(u_stat_MD, p_value_MD, "MD table Test pLDDT")


    benign_other = other_regions_table[other_regions_table["Variant Effect"] == "Benign"
    ][pLDDT_column_name].dropna()
    print(benign_other.shape, "benign pLDDT shape")
    pathogenic_other = other_regions_table[other_regions_table["Variant Effect"] == "Pathogenic"
    ][pLDDT_column_name].dropna()
    print(pathogenic_other.shape, "pathogenic pLDDT shape")

    u_stat_other, p_value_other = ss.mannwhitneyu(benign_other, pathogenic_other, alternative='two-sided')
    print(u_stat_other, p_value_other, "Others Test pLDDT")

    plt.figure(figsize=(12, 6))


    plt.subplot(1, 2, 1)

    ax1= sns.kdeplot(
        data=MD_table,
        x=GERP_column_name, y=pLDDT_column_name, hue="Variant Effect", palette=variant_palette,
        thresh=0.1, levels=5, linewidths=2, common_norm=False, legend=True
    )


    plt.xlabel("GERP++_RS")
    plt.ylabel("pLDDT")
    plt.title(title_for_MD_table)
    sns.move_legend(ax1, "upper left")
    plt.grid(True)
    if p_value_MD < 0.0001:
        plt.text(0.05, 0.05, 'pLDDT: p < 0.0001',
                ha='left', va='bottom', transform=ax1.transAxes,
                fontdict= {'size': 18})
    else:
        plt.text(0.05, 0.05, f' pLDDT: p={p_value_MD:.2f}',
                ha='left', va='bottom', transform=ax1.transAxes,
                fontdict= {'size': 18})
    plt.xlim((-6, 7.5))


    plt.subplot(1, 2, 2)

    ax2 = sns.kdeplot(
        data=other_regions_table,
        x=GERP_column_name, y=pLDDT_column_name, hue="Variant Effect", 
        palette=variant_palette,
        thresh=0.1, levels=5, linewidths=2, common_norm=False, legend=True
    )

    plt.xlabel("GERP++_RS")
    plt.ylabel("pLDDT")
    plt.title("Other Protein Regions")
    sns.move_legend(ax2, "upper left")
    plt.grid(True)
    if p_value_other < 0.0001:
        plt.text(0.05, 0.05, 'pLDDT: p < 0.0001',
                ha='left', va='bottom', transform=ax2.transAxes,
                fontdict= {'size': 18})
    else:
        plt.text(0.05, 0.05, f'pLDDT: p={p_value_other:.2f}',
                ha='left', va='bottom', transform=ax2.transAxes,
                fontdict= {'size': 18})
    
    plt.xlim((-6, 7.5))
    plt.tight_layout()
    plt.savefig(os.path.join(
        results_dir,
        "GERP_RS_pLDDT_corr.png"), dpi=300, bbox_inches="tight")

    ######

def plot_variant_by_feature(IDRs_table, 
                   GERP_column_name,variant_palette, feature_name,
                   ylabel, ylim =None, figname = None,
                   results_dir = None):
    if (figname is None) or (results_dir is None):
        ValueError("Please provide valid values for the figname and results_dir to save the plot.") 
    #upper_bound_feature = np.percentile(IDRs_table[feature_name], 99.5)
    plt.figure(figsize=(8, 6))
    ax = sns.kdeplot(
    data=IDRs_table,
    x=GERP_column_name, y= feature_name, hue="Variant Effect", palette=variant_palette,
    thresh=0.1, levels=5, linewidths=2, common_norm=False, legend=True)

    plt.xlabel("GERP++_RS")
    plt.ylabel(ylabel)
    #plt.title("IDRs")
    sns.move_legend(ax, "upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.xlim((-6, 7.5))
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(
        os.path.join(results_dir,
                     figname),
                    dpi=300, bbox_inches="tight"
    )
    plt.clf()
    # ######
    # print("Spearman Rho of pLDDT by RMSF",
    #       IDRs_table.groupby("Variant Effect").apply(
    #         lambda x: ss.spearmanr(x[RMSF_column_name], x[pLDDT_column_name],
    #                                nan_policy = "omit")
    # )
    # )
    # plt.figure(figsize=(10,5))
    # ax_3 = sns.jointplot(
    #     data=IDRs_table, x=RMSF_column_name, y=pLDDT_column_name, hue="Variant Effect",
    #     kind = "kde",
    #     marginal_kws = {"common_norm": False},
    #     joint_kws= {"common_norm": False}
    #     )
    # plt.xlabel("RMSF")
    # plt.ylabel("pLDDT")
    # ax_3.figure.suptitle("IDRs")
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig(
    #     os.path.join(results_dir,
    #                  "RMSF_pLDDT_IDRs.png"),
    #                  dpi=300, bbox_inches="tight"
    #)
    



def plot_MSA_coevolution(IDRs_table, other_regions_table,
                         site_evolution_column_name,
                         coevolution_column_name, results_dir):
    print(IDRs_table.shape, "IDRs shape coevol")
    print(IDRs_table["Variant Effect"].value_counts())
    
    print(other_regions_table.shape, "Other Regions Shape Coevol")
    print(other_regions_table["Variant Effect"].value_counts())
    ax1 = sns.jointplot(
    data=IDRs_table,
    x= site_evolution_column_name,
    y=coevolution_column_name, 
    hue="Variant Effect",  
    kind="kde",  
    thresh=0.1, 
    levels=5,
    fill=False,  
    gridsize=2000,
    clip=((-0.5, 1.5), (-0.5, 1.5)),
    joint_kws= {"common_norm": False},
    marginal_kws={"common_norm": False}
    )
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    ax1.figure.suptitle("IDRs")

    sns.move_legend(ax1.ax_joint, "upper right")
    plt.grid(True)

    sns.regplot(
        x=site_evolution_column_name, y=coevolution_column_name, 
        data=IDRs_table[IDRs_table["Variant Effect"] == "Benign"], ax=ax1.ax_joint, 
        scatter=False, ci=95, 
        line_kws= {'linestyle': '--', 'alpha': 0.5}
    )

    plt.xlabel("Site Specific Entropy")
    plt.ylabel("Maximum MI")

    plt.tight_layout()
    plt.savefig(os.path.join(
        results_dir,"Co-evolution_Variant_Type_MI_IDRs.png"),
          dpi=300, bbox_inches="tight")
    

    ax2 = sns.jointplot(
    data=other_regions_table,
    x=site_evolution_column_name, 
    y=coevolution_column_name, 
    hue="Variant Effect",  
    kind="kde",  
    thresh=0.1, 
    levels=5,
    fill=False,  
    gridsize=2000,
    clip=((-0.5, 1.5), (-0.5, 1.5)),
    joint_kws= {"common_norm": False},
    marginal_kws={"common_norm": False}
)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    
    ax2.figure.suptitle("Other Protein Regions")

    sns.move_legend(ax2.ax_joint, "upper right")
    plt.grid(True)
    sns.regplot(
        x=site_evolution_column_name, y=coevolution_column_name, 
        data=other_regions_table[other_regions_table["Variant Effect"] == "Benign"], ax=ax2.ax_joint, 
        scatter=False, ci=95, 
        line_kws= {'linestyle': '--', 'alpha': 0.5}
    )

    plt.xlabel("Site Specific Entropy")
    plt.ylabel("Maximum MI")
    plt.tight_layout()
    plt.savefig(os.path.join(
                results_dir, "Co-evolution_Variant_Type_MI_others.png"),
                  dpi=300, bbox_inches="tight")
    ######


    benign_data_IDRs = IDRs_table[
    IDRs_table["Variant Effect"] == "Benign"
    ]

    X_benign = benign_data_IDRs[[coevolution_column_name]]
    y_benign = benign_data_IDRs[site_evolution_column_name]

    model = LinearRegression()
    model.fit(X_benign, y_benign)

    pathogenic_data_IDRs = IDRs_table[
        IDRs_table["Variant Effect"] == "Pathogenic"
    ]

    X_pathogenic = pathogenic_data_IDRs[[coevolution_column_name]]

    predicted_coevol = model.predict(X_pathogenic)

    pathogenic_data_IDRs["Predicted_coevol"] = predicted_coevol

    pathogenic_data_IDRs["Residuals"] = pathogenic_data_IDRs[coevolution_column_name] - pathogenic_data_IDRs["Predicted_coevol"]


    print("t-test IDRs", ss.ttest_1samp(pathogenic_data_IDRs["Residuals"], 0, alternative = "two-sided"))

    #####

    benign_data_others = other_regions_table[
        other_regions_table["Variant Effect"] == "Benign"
    ]

    X_benign = benign_data_others[[site_evolution_column_name]]
    y_benign = benign_data_others[coevolution_column_name]

    model = LinearRegression()
    model.fit(X_benign, y_benign)

    pathogenic_data_others = other_regions_table[
        other_regions_table["Variant Effect"] == "Pathogenic"
    ]

    X_pathogenic = pathogenic_data_others[["Site_Specific_Entropy"]]

    predicted_Max_MI = model.predict(X_pathogenic)

    pathogenic_data_others["Predicted_coevol"] = predicted_Max_MI

    pathogenic_data_others["Residuals"] = pathogenic_data_others[coevolution_column_name] - pathogenic_data_others["Predicted_coevol"]


    ##########
    
    ax1 = sns.jointplot(
        data=pathogenic_data_IDRs, 
        x=site_evolution_column_name, 
        y="Residuals",
        marginal_kws= {"common_norm":False} 
    )
    ax1.figure.suptitle('IDRs')
    ax1.set_xlabel('Site Specific Entropy')
    ax1.set_ylabel('Residuals of Coevolution')
    ax1.axhline(0, color='black', linestyle='--')  
    ax1.figure.savefig(
        os.path.join(results_dir, "IDRs_coevolution_residuals.png",
                    bbox_inches = "tight",
                     dpi=300)
    )
    plt.clf()
    ax2 = sns.jointplot(
        data=pathogenic_data_others, 
        x="Site_Specific_Entropy", 
        y="Residuals", 
        marginal_kws= {"common_norm":False}, 
    )
    ax2.figure.suptitle('Other Protein Regions')
    ax2.set_xlabel('Site Specific Entropy')
    ax2.set_ylabel('Residuals for Max MI')
    ax2.axhline(0, color='black', linestyle='--') 

    ax2.figure.savefig(
        os.path.join(results_dir,
                     "Other_regions_coevolution_residuals.png"), dpi=300, bbox_inches="tight")


def plot_MSA_features_RMSF(IDRs_table,
                         site_evolution_column_name,
                         coevolution_column_name,
                         RMSF_column_name, results_dir):
    

    ax1 = sns.jointplot(
    data=IDRs_table,
    x= site_evolution_column_name,
    y= RMSF_column_name, 
    hue="Variant Effect",  
    kind="kde",  
    thresh=0.1, 
    levels=5,
    fill=False,  
    gridsize=2000,
    joint_kws= {"common_norm": False},
    marginal_kws={"common_norm": False}
    )
    plt.xlim(-0.5, 1.5)
    plt.xlabel("Site Specific Entropy")
    plt.ylabel("RMSF")
    ax1.figure.suptitle("IDRs")

    sns.move_legend(ax1.ax_joint, "upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(
        results_dir,"Evolution_RMSF_IDRs.png"),
          dpi=300, bbox_inches="tight")
    
    ax2 = sns.jointplot(
    data=IDRs_table,
    x= coevolution_column_name,
    y= RMSF_column_name, 
    hue="Variant Effect",  
    kind="kde",  
    thresh=0.1, 
    levels=5,
    fill=False,  
    gridsize=2000,
    joint_kws= {"common_norm": False},
    marginal_kws={"common_norm": False}
    )
    plt.xlim(-0.5, 1.5)
  
    plt.xlabel("Max MI")
    plt.ylabel("RMSF")
    ax2.figure.suptitle("IDRs")

    sns.move_legend(ax2.ax_joint, "upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(
        results_dir, "Coevolution_RMSF_IDRs.png"),
          dpi=300, bbox_inches="tight")
    
    #########

    IDRs_table = IDRs_table[IDRs_table["Variant Effect"] == "Pathogenic"]
    ax1 = sns.jointplot(
    data=IDRs_table,
    x= site_evolution_column_name,
    y= RMSF_column_name, 
    kind="reg"
    )

    plt.xlabel("Site Specific Entropy")
    plt.ylabel("RMSF")
    ax1.figure.suptitle("IDRs")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(
        results_dir,"Evolution_RMSF_IDRs_pathogenic.png"),
          dpi=300, bbox_inches="tight")
    
    ax2 = sns.jointplot(
    data=IDRs_table,
    x= coevolution_column_name,
    y= RMSF_column_name, 
    kind="reg"

    )
    
    plt.xlabel("Max MI")
    plt.ylabel("RMSF")
    ax2.figure.suptitle("IDRs")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(
        results_dir, "Coevolution_RMSF_IDRs_pathogenic.png"),
          dpi=300, bbox_inches="tight")




def plot_DSSP(MD_table, structured_dssp_column_names,
              unstructured_dssp_column_names,
              MD_feature_table_title,
              variant_palette,
              Length_column_name,
              Length_cutoff, 
              RMSF_column_name, 
              RMSF_cutoff,
              pLDDT_column_name,
              pLDDT_cutoff,
              significance_levels,
              results_dir):
    
    matplotlib.rcParams.update({'font.size': 16})

    #IDRs_table[dssp_column_names] = IDRs_table[dssp_column_names].applymap(lambda x: np.log(x / (1 - x + 1e-6) ))
    #Logit transform to enlarge small proportion values
    for dssp_columns, dssp_labels, label in [
        (list(structured_dssp_column_names.keys()),
         list(structured_dssp_column_names.values()), "Structured"),
        (list(unstructured_dssp_column_names.keys()),
         list(unstructured_dssp_column_names.values()), "Unstructured")]:
        
        # Prepare data
        MD_table.rename(columns = structured_dssp_column_names, inplace=True)
        MD_table.rename(columns = unstructured_dssp_column_names, inplace = True)
        MD_res_features = MD_table[dssp_labels+ [Length_column_name,
                                                       RMSF_column_name,
                                                       pLDDT_column_name,
                                                       "Variant Effect"]].melt(
            id_vars=["Variant Effect", Length_column_name, RMSF_column_name, pLDDT_column_name],
            var_name="DSSP", value_name="Proportion"
        )
        
        plt.figure(figsize=(8, 6))
        sns.barplot(data=MD_res_features, x='DSSP', y='Proportion', 
                    hue='Variant Effect', palette=variant_palette,ci=68)
        plt.xlabel("DSSP")
        plt.ylabel("Proportion of DSSP Assignment")
        plt.title(f"{label} Secondary Structure and Variant Effect ({MD_feature_table_title})")
        plt.xticks(rotation = 45, ha = "right",fontsize = 16)

        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{label}_DSSP_variant_type.png"), dpi=300, bbox_inches="tight")
        plt.clf()

        #### RMSF/pLDDT Categories

        MD_res_features_pathogenic = MD_res_features[MD_res_features["Variant Effect"] == "Pathogenic"]

        if Length_cutoff is not None:
            MD_res_features_pathogenic['Length Category'] = np.select(
                [
                    MD_res_features_pathogenic[Length_column_name] > Length_cutoff,
                    MD_res_features_pathogenic[Length_column_name] <= Length_cutoff
                ],
                ['Long IDRs', 'Short IDRs']
            )
            palette = {"Long IDRs": "#e81a1a", "Short IDRs": "#f5ed11"}

            #Plot for RMSF/pLDDT categories
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=MD_res_features_pathogenic, x='DSSP', y='Proportion', 
                        hue='Length Category', ci=68, palette=palette)
            plt.xlabel("DSSP")
            plt.ylabel("Proportion of DSSP Assignment")
            plt.title(f"{label} DSSP Assignments (Length Cutoff: {Length_cutoff})")
            plt.xticks(rotation = 45, fontsize = 16)
            plt.grid(True)
            ax.set_axisbelow(False)

            num_tests = len(dssp_labels)
            for i, dssp_col in enumerate(dssp_labels):
                long_values = MD_res_features_pathogenic.loc[
                    (MD_res_features_pathogenic['Length Category'] == 'Long IDRs') &
                    (MD_res_features_pathogenic['DSSP'] == dssp_col)]["Proportion"]
                short_values = MD_res_features_pathogenic.loc[
                    (MD_res_features_pathogenic['Length Category'] == 'Short IDRs') &
                    (MD_res_features_pathogenic['DSSP'] == dssp_col)]["Proportion"]

                if len(long_values) > 0 and len(short_values) > 0:
                    stat, p_value = ss.mannwhitneyu(long_values, short_values, alternative='two-sided',nan_policy = "omit")
                    adjusted_p = min(p_value * num_tests, 1) #bonferroni correction
                    sig_symbol = next((symbol for threshold, symbol in significance_levels.items() if adjusted_p < threshold), '')
                    if sig_symbol:
                        
                        y_max = long_values.mean() *1.02
                        ax.text(i, y_max, sig_symbol, ha='center', fontsize=16, color='black')

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{label}_DSSP_Length_categories.png"),
                        dpi=300, bbox_inches="tight")
            plt.clf()
        

        if RMSF_cutoff is not None:
            MD_res_features_pathogenic['RMSF Category'] = np.select(
                [
                    MD_res_features_pathogenic[RMSF_column_name] > RMSF_cutoff,
                    MD_res_features_pathogenic[RMSF_column_name] <= RMSF_cutoff
                ],
                ['High RMSF', 'Low RMSF']
            )
            palette = {"High RMSF": "#e81a1a", "Low RMSF": "#f5ed11"}

            #Plot for RMSF/pLDDT categories
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=MD_res_features_pathogenic, x='DSSP', y='Proportion', 
                        hue='RMSF Category', ci=68)
            plt.xlabel("DSSP")
            plt.ylabel("Proportion of DSSP Assignment")
            plt.title(f"{label} DSSP Assignments (RMSF Cutoff: {RMSF_cutoff})")
            plt.xticks(ticks=np.arange(len(dssp_columns)), labels=dssp_labels,
                    rotation=45, ha="right")
            plt.grid(True)
            ax.set_axisbelow(False)

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{label}_DSSP_RMSF_categories.png"),
                        dpi=300, bbox_inches="tight")
            plt.clf()
        elif pLDDT_cutoff is not None:
            MD_res_features_pathogenic['pLDDT Category'] = np.select(
                [
                    MD_res_features_pathogenic[pLDDT_column_name] > pLDDT_cutoff,
                    MD_res_features_pathogenic[pLDDT_column_name] <= pLDDT_cutoff
                ],
                ['High pLDDT', 'Low pLDDT']
            )
            #Plot for RMSF/pLDDT categories
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=MD_res_features_pathogenic, x='DSSP', y='Proportion', 
                        hue='pLDDT Category', ci=68)
            plt.xlabel("DSSP")
            plt.ylabel("Proportion of DSSP Assignment")
            plt.title(f"{label} DSSP Assignments (pLDDT Cutoff: {pLDDT_cutoff})")
            plt.xticks(ticks=np.arange(len(dssp_columns)), labels=dssp_labels,
                    rotation=45, ha="right")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{label}_DSSP_pLDDT_categories.png"),
                        dpi=300, bbox_inches="tight")
            plt.clf()



def plot_sequence_composition(
              IDRs_table, 
              group_col,
              palette,
              sasa_average_column,
              sequence_composition_column,
              average_conservation_column,
              average_gamma_column,
              std_gamma_column,
              original_aa_column,
              changed_aa_column,
              results_dir):
    
    
    
    IDRs_table[sequence_composition_column] = np.arcsin(np.sqrt(IDRs_table[sequence_composition_column]))
    IDRs_table[average_gamma_column] = np.log10(IDRs_table[average_gamma_column] + 1e-4)

    IDRs_table[std_gamma_column] = np.log10(IDRs_table[std_gamma_column] + 1e-4)
        
    
    charged_residues = {"R", "H", "K", "D", "E"}

    def classify_mutation_type_charged(row):
        original_in_charged = row[original_aa_column] in charged_residues
        changed_in_charged = row[changed_aa_column] in charged_residues
        
        if original_in_charged and changed_in_charged:
            return "Charged --> Charged"
        elif original_in_charged and not changed_in_charged:
            return "Charged --> Uncharged"
        elif not original_in_charged and changed_in_charged:
            return "Uncharged --> Charged"
        else:
            return "Uncharged --> Uncharged"
        
    positive_residues = {"R", "H", "K"}
    negative_residues = {"D", "E"}

    def classify_orig_aa_charged(row):
        original_in_positive = row[original_aa_column] in positive_residues
        original_in_negative = row[original_aa_column] in negative_residues
        if original_in_positive:
            return "Positive"
        elif original_in_negative:
            return "Negative"
        else:
            return "Uncharged"


    IDRs_table['mutation_type_charge'] = IDRs_table.apply(classify_mutation_type_charged, axis=1)
    contingency_table = pd.crosstab(IDRs_table['mutation_type_charge'], IDRs_table[group_col])
    print(contingency_table)
    print(ss.chi2_contingency(contingency_table))


    plot_boxplot_with_significance(
        data=IDRs_table,
        group_col=group_col,
        value_col=sequence_composition_column,
        results_dir=results_dir,
        xlabel = group_col, 
        ylabel = "Conservation of\nSequence Composition",
        plot_filename=f"sequence_composition_{group_col}.png",
        palette = palette,
        verbose = True,
        plot_type = "violin"
    )
    plt.clf()
    plot_boxplot_with_significance(
        data=IDRs_table,
        group_col=group_col,
        value_col=average_conservation_column,
        results_dir=results_dir,
        xlabel = group_col, 
        ylabel = "Average Entropy of Sequence Region",
        plot_filename=f"average_conservation_{group_col}.png",
        palette = palette

        
    )
    plt.clf()
    plot_boxplot_with_significance(
        data=IDRs_table,
        group_col=group_col,
        value_col=average_gamma_column,
        results_dir=results_dir,
        xlabel = group_col, 
        ylabel = "Charge Segregation",
        plot_filename=f"average_gamma_{group_col}.png",
        palette = palette, 
        bar_height = 0.6

    )
    plt.clf()
    plot_boxplot_with_significance(
        data=IDRs_table,
        group_col=group_col,
        value_col=std_gamma_column,
        results_dir=results_dir,
        xlabel = group_col, 
        ylabel = "Variation in Charge Segregation (Homologs)",
        plot_filename=f"stdev_gamma_{group_col}.png",
        bar_height = 0.5,
        palette = palette
    )
    plt.clf()
    
    plot_boxplot_with_significance(
        data=IDRs_table,
        group_col=group_col,
        value_col=sasa_average_column,
        results_dir=results_dir,
        xlabel = group_col, 
        ylabel = "Average Surface\nArea (nm)",
        plot_filename=f"sasa_average_{group_col}.png",
        bar_height= 0.5,
        palette = palette
    )
    plt.clf()

    # pairwise_columns = [
    #     average_conservation_column,
    #     sequence_composition_column,
    #     std_gamma_column
    # ]

    # sns.pairplot(
    #     IDRs_table[IDRs_table["RMSF Category"] != "Benign"],
    #     vars=pairwise_columns,
    #     kind="reg",
    #     hue = "RMSF Category",
    #     diag_kind="kde",
    #     plot_kws={'scatter_kws': {'s': 10}}
    # )
    # plt.suptitle("IDRs", y=1.02)
    # plt.savefig(os.path.join(results_dir, "pairwise_reg_physical_features.png"), dpi=300, bbox_inches="tight")
    # plt.clf()

    # sns.scatterplot(
    #     IDRs_table[IDRs_table["Variant Effect"]=="Benign"],
    #     x = RMSF_column_name,
    #     y = std_gamma_column,
    #     s = 10,
    #     alpha = 0.4)
    # plt.title("Benign Variants in IDRs", y=1.02)
    # plt.xlabel("RMSF")
    # plt.ylabel("Standard Devation of Gamma")
    # plt.savefig(os.path.join(results_dir, "RMSF_std_gamma_scatter.png"), dpi=300, bbox_inches="tight")
    # plt.clf()

    # print(ss.spearmanr(IDRs_table[IDRs_table["Variant Effect"] == "Benign"][RMSF_column_name],
    #       IDRs_table[IDRs_table["Variant Effect"] == "Benign"][std_gamma_column]) )


    # IDRs_table['Residue Charge'] = IDRs_table.apply(classify_orig_aa_charged, axis=1)
    

    # plot_boxplot_with_significance(
    #     data=IDRs_table,
    #     group_col= "Residue Charge",
    #     value_col=RMSF_column_name,
    #     results_dir=results_dir,
    #     ylabel = "RMSF",
    #     plot_filename="RMSF_Charge.png",
    #     plot_type = "violin"
    # )
    # plt.clf()



def plot_GO_terms(IDRs_table_GO, 
                  GO_column_name,
                  top_n,
                  RMSF_column_name, 
                  RMSF_cutoff,
                  results_dir):
    IDRs_table_GO['RMSF Category'] = np.select(
        [
            (IDRs_table_GO["Variant Effect"] == "Pathogenic") & 
            (IDRs_table_GO[RMSF_column_name] > RMSF_cutoff),
            (IDRs_table_GO["Variant Effect"] == "Pathogenic") &
            (IDRs_table_GO[RMSF_column_name] <= RMSF_cutoff),
            IDRs_table_GO["Variant Effect"] == "Benign"
        ],
        ['Pathogenic - High RMSF', 'Pathogenic - Low RMSF', 'Benign']
    )

    top_go_terms = IDRs_table_GO[GO_column_name].value_counts().head(top_n).index
    print(top_go_terms)
    significant_associations = []

    for go_term in top_go_terms:
        IDRs_table_GO['GO Category'] = IDRs_table_GO[GO_column_name].apply(
            lambda x: go_term if x == go_term else "Other"
        )

        contingency_table = pd.crosstab(IDRs_table_GO['RMSF Category'],
                                         IDRs_table_GO["GO Category"])

        print(contingency_table)

        chi2, p, dof, _ = ss.chi2_contingency(contingency_table)

        if p < 0.05:
            significant_associations.append({
                'GO Term': go_term,
                'Chi2': chi2,
                'p-value': p,
                'Degrees of Freedom': dof
            })

    results_df = pd.DataFrame(significant_associations)
    
    results_df.sort_values(by='p-value', inplace=True)
    print(results_df)




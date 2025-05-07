import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as ss
import numpy as np
import os
import warnings

import sys
import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.append(ROOT)
from utils import *
from config import config
from plotter_functions import *

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.rcParams.update({'font.size': 13})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)




def main():
    data_dir = os.path.abspath(config["data_dir"])
    results_dir = os.path.join(os.path.abspath(config["results_dir"]), "clinical_figures") #recommend adding subdirectories to save specific figures
    
    
    
    IDR_MD_features = pd.read_csv(
        os.path.join(
            data_dir, "clinical_train_val", "feature_table.csv"
        ) 
        , index_col= 0, low_memory= False
    )
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
                          how="left", suffixes=('', '_y'))
    
    #print(IDRs_table.shape)
    other_regions_table = pd.merge(IDR_MD_features, 
                          proteome_information,
                          left_on=["UniProtID", location_column_name, "Changed AA"],
                          right_on=["UniProtID", location_column_name, "changed_aa_amis"],
                          how="outer", suffixes=('_x', ''), indicator=True)
    
    other_regions_table = other_regions_table[
        other_regions_table["_merge"] == "right_only"].drop(columns=["_merge"])
    
    #print(other_regions_table["outcome"].isna().sum())
    other_regions_table[outcome_column_name] = other_regions_table[outcome_column_name].astype(int)
    
    
    IDRs_table = IDRs_table[IDRs_table["Label Source"]!= "HGMD"] # not used for plotting either, but for removing IDRs from other protein regions 

    #print(IDRs_table["outcome"].value_counts())
    #print(other_regions_table["outcome"].value_counts())

    IDRs_table["Variant Effect"] = np.where(IDRs_table[outcome_column_name] == 1, "Pathogenic", "Benign")
    other_regions_table["Variant Effect"] = np.where(other_regions_table[outcome_column_name] == 1, "Pathogenic", "Benign")


    
    sasa_column_name = "Res_MD_1_pos_0"
    RMSF_column_name = "Res_MD_3_pos_0"
    pLDDT_column_name = "pLDDT"
    GERP_column_name = "GERP++_RS"


    IDRs_table["start"] = IDRs_table["protein_start_end"].str.split("_").str[1].astype(int)
    IDRs_table["end"] = IDRs_table["protein_start_end"].str.split("_").str[2].astype(int)

    IDRs_table["Region Length"] = IDRs_table["end"] - IDRs_table["start"] + 1



    #Trying by removing the start loss mutations
    IDRs_table = IDRs_table[IDRs_table[location_column_name] !=1]


    IDRs_table['RMSF Category'] = np.select(
        [
            (IDRs_table["Variant Effect"] == "Pathogenic") & 
            (IDRs_table[RMSF_column_name] > 4.5),
            (IDRs_table["Variant Effect"] == "Pathogenic") &
            (IDRs_table[RMSF_column_name] <= 4.5),
            IDRs_table["Variant Effect"] == "Benign"
        ],
        ['Pathogenic - High RMSF', 'Pathogenic - Low RMSF', 'Benign']
    )

    IDRs_table['Length Category'] = np.select(
        [
            (IDRs_table["Variant Effect"] == "Pathogenic") & 
            (IDRs_table["Region Length"] > 800),
            (IDRs_table["Variant Effect"] == "Pathogenic") &
            (IDRs_table["Region Length"] <= 800),
            IDRs_table["Variant Effect"] == "Benign"
        ],
        ['Pathogenic - Long IDRs', 'Pathogenic - Short IDRs', 'Benign']
    )


    #print(IDRs_table["Length Category"].value_counts())

    #print(IDRs_table[IDRs_table["Length Category"]== "Pathogenic - Long IDRs"]["protein_start_end"].value_counts())
    #print(IDRs_table[IDRs_table["Length Category"]== "Pathogenic - Long IDRs"]["protein_start_end"].nunique(), "Unique Long IDRs")



    #### Plotting begins below!
    length_palette = {"Pathogenic - Long IDRs": "#e81a1a", "Pathogenic - Short IDRs": "#f5ed11",
                      "Benign": "#3497ed"}
    variant_palette = {"Pathogenic": "#ffa500", "Benign": "#3497ed"}


    #### Comparing ESM1b with AlphaMissense
    print(ss.spearmanr(IDRs_table["ESM_probabilities"],
                       IDRs_table["am_pathogenicity"], nan_policy="omit"),
                       "Corr between ESM and AM")
    plt.figure(figsize=(10,10))
    g = sns.jointplot(data= IDRs_table, x="ESM_probabilities",
                    y = "am_pathogenicity", hue="Length Category",
                    palette = length_palette,
                    marginal_kws={'common_norm': False,  
                             'alpha': 0.7},
                    joint_kws = {'alpha':0.5})
    g.ax_joint.legend_.set_title(None)
    sns.move_legend(g.figure.axes[0], loc='upper left', bbox_to_anchor=(1.19, 1),
                    labels = ["Benign", "Pathogenic\nShort IDRs", "Pathogenic\nLong IDRs"])

    plt.xlim(0.2,0.8)
    plt.ylim(0,1)
    plt.ylabel("AlphaMissense Score", fontdict={"size":18})
    plt.xlabel("ESM1b Probability", fontdict={"size":18})
    plt.savefig(os.path.join(results_dir, "IDR_ESM1b_AM_corr.png"),
                dpi = 300, bbox_inches = "tight")
    plt.clf()
    
    ###

    plot_GERP_info(IDRs_table, other_regions_table, GERP_column_name,
                   "am_pathogenicity", "Variant Effect", variant_palette,
                  results_dir)
    plot_pLDDT_info(IDRs_table, other_regions_table, "IDRs", 
                    variant_palette, GERP_column_name,
                    pLDDT_column_name,
                  results_dir)
    
    ## Plot secondary structure
    structured_dssp_column_names = {"Res_MD_4_pos_0": 'Beta Bridge',
                                     "Res_MD_5_pos_0": 'Extended Strand',
                                     "Res_MD_6_pos_0":'Alpha Helix',
                                     "Res_MD_7_pos_0": '5-helix',
                                    "Res_MD_8_pos_0": '3-helix'}
    unstructured_dssp_column_names = {"Res_MD_9_pos_0":'Bend',
                                      "Res_MD_10_pos_0": 'Hydrogen Bonded Turn',
                                      "Res_MD_11_pos_0": 'Loops and Irregular Elements'}
    
    
    plot_DSSP(IDRs_table, structured_dssp_column_names,
              unstructured_dssp_column_names, "IDRs",
              variant_palette,
              "Region Length", 800,
              RMSF_column_name, None, pLDDT_column_name, None, 
              significance_levels = {0.0001: '***', 0.001: '**', 0.05: '*'},
              results_dir = results_dir)
    
    
    ## Bringing in Sequence/MSA features
    sequence_composition_df = pd.read_csv(
        os.path.join(data_dir, "sequence_composition_MSA.csv"), index_col = 0
    )

    sequence_composition_df.drop_duplicates(subset=["UniProtID", "Location"],inplace=True)
    
    IDRs_table_with_composition = pd.merge(left = IDRs_table, right = sequence_composition_df,
                                           left_on= ["UniProtID", "location"],
                                           right_on= ["UniProtID", "Location"],
                                           how = "inner")
    print(IDRs_table_with_composition.shape)

    print(IDRs_table_with_composition["Length Category"].value_counts())
    
    plot_sequence_composition(IDRs_table_with_composition, 
                            "Length Category",
                            length_palette,
                            sasa_column_name,
                            "Average Cosine Similarity",
                            "Average Entropy Region",
                            "Average Gamma",
                            "StDev Gamma",
                            "Original AA",
                            "Changed AA",
                            results_dir
                            )
    conformational_properties = pd.read_csv(os.path.join(
        data_dir, "conformational_properties.csv"
    ))
    IDRs_table_with_conf_prop = pd.merge(left = IDRs_table_with_composition, right = conformational_properties,
                                           left_on= "protein_start_end",
                                           right_on="seq_name",
                                           how="left")
    print(IDRs_table_with_conf_prop.shape, "With conformation")
    


    sns.regplot(data= IDRs_table_with_conf_prop, x="Region Length",
                    y = RMSF_column_name)
    plt.ylabel("RMSF")
    plt.savefig(os.path.join(results_dir, "IDR_Length_RMSF_scatter.png"),
                dpi = 300, bbox_inches = "tight")
    plt.clf()

    sns.regplot(data= IDRs_table_with_conf_prop, x="Region Length",
                    y = "Average RMSF")
    plt.ylabel("Average RMSF of IDR")
    plt.savefig(os.path.join(results_dir, "IDR_Length_Avg_RMSF_scatter.png"),
                dpi = 300, bbox_inches = "tight")
    plt.clf()

    sns.regplot(data= IDRs_table_with_conf_prop, x="Region Length",
                    y = "Average SASA")
    plt.ylabel("Average SASA of IDR")
    plt.savefig(os.path.join(results_dir, "IDR_Length_Avg_SASA_scatter.png"),
                dpi = 300, bbox_inches = "tight")
    plt.clf()


    plot_variant_by_feature(IDRs_table_with_conf_prop, 
                            GERP_column_name,variant_palette, RMSF_column_name, "Root-Mean-Square Fluctuation",
                            (0,1.5),
                            "RMSF_Variant.png",
                            results_dir)
    
    # plot_variant_by_feature(IDRs_table_with_conf_prop, 
    #                         GERP_column_name,variant_palette, sasa_column_name, "SASA",
    #                         "SASA_Variant.png",
    #                results_dir)
    
    
    plot_variant_by_feature(IDRs_table_with_conf_prop, 
                            GERP_column_name,variant_palette, "Region Length", "Region Length",
                            None,
                            "Length_Variant.png",
                   results_dir)
    
    # plot_variant_by_feature(IDRs_table_with_conf_prop, 
    #                         GERP_column_name,variant_palette, "Pair_MD_1", "Average Van Der Waal's Forces",
    #                         "Avg_VDW.png",
    #                results_dir)
    
    # plot_variant_by_feature(IDRs_table_with_conf_prop, 
    #                         GERP_column_name,variant_palette, "Pair_MD_2", "Average Hydrogen Bonds (BB-BB)",
    #                         "Avg_HBBB.png",
    #                results_dir)
    
    plot_ridgeplot(
        data=IDRs_table,
        group_col='Length Category',
        value_col="Pair_MD_1",
        results_dir=results_dir,
        xlabel = "Number of Van Der Waal's Interactions",
        plot_filename="VDW_Length_Category.png",
        label_placement ="right", palette = length_palette, 
        xlim = (0,25), bar_height = 0.05, xlim_buffer = 1.025
    )
    plt.clf()

    plot_ridgeplot(
        data=IDRs_table_with_composition,
        group_col='Length Category',
        value_col="Average Entropy Site",
        results_dir=results_dir,
        xlabel = "Site Specific Entropy",
        plot_filename="Site_Entropy.png", palette = length_palette,label_placement = "right",
        xlim =(-0.1, 2.5),
        bar_height = 0.05, xlim_buffer = 1.025
    )
    plt.clf()

    plot_ridgeplot(
        data=IDRs_table_with_composition,
        group_col='Length Category',
        value_col="Average Gamma",
        results_dir=results_dir,
        xlabel = "Average Charge Pattern",
        plot_filename="average_gamma_length.png", palette = length_palette,label_placement = "right",
        xlim = (-0.1, 20), bar_height = 0.05, xlim_buffer = 1.025
    )
    plt.clf()

    plot_ridgeplot(
        data=IDRs_table_with_composition,
        group_col='Length Category',
        value_col="Probability of STY",
        results_dir=results_dir,
        xlabel = "Probability of STY",
        plot_filename="sty_length.png", palette = length_palette,label_placement = "right",
         bar_height = 0.05, xlim_buffer = 1.025
    )
    plt.clf()

    plot_ridgeplot(
        data=IDRs_table_with_composition,
        group_col='Length Category',
        value_col="Average Entropy Region",
        results_dir=results_dir,
        xlabel = "Average Entropy of Window",
        plot_filename="Region_Entropy.png", palette = length_palette,label_placement = "right",
        xlim = (-0.1, 2.5), bar_height = 0.05, xlim_buffer = 1.025
    )
    plt.clf()

    plot_ridgeplot(
        data=IDRs_table_with_conf_prop,
        group_col='Length Category',
        value_col="nu",
        results_dir=results_dir,
        xlabel = "Length Adjusted Compaction (nu)",
        plot_filename="nu_Length_Category.png",
        label_placement ="left", palette = length_palette, 
        xlim = (0.3,0.7), bar_height = 0.05, xlim_buffer = 1.025
    )
    plt.clf()

    plot_ridgeplot(
        data=IDRs_table,
        group_col='Length Category',
        value_col="Pair_MD_2",
        results_dir=results_dir,
        xlabel = "Number of Hydrogen Bonds (BB-BB)",
        plot_filename="HBBB_Length_Category.png", palette = length_palette,label_placement = "right",
        xlim = (-0.5,6), bar_height = 0.05, xlim_buffer = 1.025, type = "bar"
    )
    plt.clf()
    
    plot_boxplot_with_significance(
        data=IDRs_table,
        group_col='Length Category',
        value_col="Pair_MD_10",
        results_dir=results_dir,
        ylabel = "Salt Bridges",
        plot_filename="SB_Length_Category.png",
        plot_type = "violin", bar_height = 2
    )
    plt.clf()


    print(IDRs_table_with_conf_prop.drop_duplicates("protein_start_end")["protein_start_end"].nunique(), "Unique IDRs")
    print(IDRs_table_with_conf_prop["UniProtID"].nunique(), "Unique proteins")

    ax = sns.histplot(IDRs_table_with_conf_prop.drop_duplicates("protein_start_end"), 
                 x="Region Length", bins=40)
    plt.title("Unique protein regions", fontsize =25)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('Region Length', fontsize=25)
    ax.set_ylabel('Observations', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "regions_length_histogram.png"), dpi=300, bbox_inches="tight")
    plt.clf()

    ## Switching to GPCRmd
    results_dir = os.path.join(os.path.abspath(config["results_dir"]), "GPCR_figures")
    
    train_feature_table = pd.read_csv(
        os.path.join(data_dir, "GPCRmd_train_val", "train.csv"), 
        index_col = 0
    )
    val_feature_table = pd.read_csv(
        os.path.join(data_dir, "GPCRmd_train_val", "val.csv"), 
        index_col = 0
    )

    GPCRmd_features = pd.concat([train_feature_table, val_feature_table],
                                axis = 0)
    
    proteome_information = pd.read_csv(
        os.path.join(data_dir, "merged_proteome_information_clinical.csv"),
        index_col = 0, low_memory= False
    )

    print(proteome_information["outcome"].value_counts())

    print(GPCRmd_features.shape)
    #print(GPCRmd_features.head())
    GPCRs_table = pd.merge(GPCRmd_features, 
                          proteome_information,
                          left_on=["UniProtID", "location", "Changed AA"],
                          right_on=["UniProtID", "location", "changed_aa_amis"],
                          how="left", suffixes=('', '_y'))
    print(GPCRs_table.shape)
    other_regions_table = pd.merge(GPCRmd_features, 
                          other_regions_table,
                          left_on=["UniProtID", "location", "Changed AA"],
                          right_on=["UniProtID", "location", "changed_aa_amis"],
                          how="outer", suffixes=('_x', ''), indicator=True)
    
    other_regions_table = other_regions_table[
        other_regions_table["_merge"] == "right_only"].drop(columns=["_merge"])
    
    other_regions_table["outcome"] = other_regions_table["outcome"].astype(int)
    
    print(GPCRs_table["outcome"].value_counts())
    print(other_regions_table["outcome"].value_counts())

    GPCRs_table["Variant Effect"] = np.where(GPCRs_table["outcome"] == 1, "Pathogenic", "Benign")
    other_regions_table["Variant Effect"] = np.where(other_regions_table["outcome"] == 1, "Pathogenic", "Benign")
    
    RMSF_column_name = "Res_MD_3"
    pLDDT_column_name = "pLDDT"


    GPCRs_table["start"] = GPCRs_table["MD_protein_start_end"].str.split("_").str[1].astype(int)
    GPCRs_table["end"] = GPCRs_table["MD_protein_start_end"].str.split("_").str[2].astype(int)

    GPCRs_table["Region Length"] = GPCRs_table["end"] - GPCRs_table["start"]
    print("Average Length", np.nanmean(GPCRs_table["Region Length"]))
    print("Max Length", np.nanmax(GPCRs_table["Region Length"]))

    #Remove the RMSF outlier
    GPCRs_table = GPCRs_table[GPCRs_table["Res_MD_3"] <4]
    plot_pLDDT_info(GPCRs_table, other_regions_table, "GPCRs", variant_palette,
                    "GERP++_RS",
                    pLDDT_column_name,
                   results_dir)
    
    structured_dssp_column_names = {"Res_MD_4": 'Beta Bridge',
                                     "Res_MD_5": 'Extended Strand',
                                     "Res_MD_6":'Alpha Helix',
                                     "Res_MD_7": '5-helix',
                                    "Res_MD_8": '3-helix'}
    unstructured_dssp_column_names = {"Res_MD_9": 'Bend',
                                     "Res_MD_10": 'Hydrogen Bonded Turn',
                                     "Res_MD_11":'Loops and Irregular Elements'}
    plot_DSSP(GPCRs_table, structured_dssp_column_names,
              unstructured_dssp_column_names, "GPCRs",variant_palette,
              "Region Length", None,
              RMSF_column_name, None, pLDDT_column_name, None, 
              significance_levels = {0.0001: '***', 0.001: '**', 0.05: '*'},
              results_dir = results_dir)
 
    plot_boxplot_with_significance(
        data=GPCRs_table,
        group_col='Variant Effect',
        value_col="Res_MD_1",
        results_dir=results_dir,
        palette = variant_palette,
        xlabel = "Variant Effect",
        ylabel = "Solvent Accessible\nSurface Area (nm)",
        plot_filename="SASA_GPCRs.png",
        plot_type = "box", bar_height = 0.5
    )

if __name__ == "__main__":
    main()

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

import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/az2798/MDmis/code/')

from utils import *
warnings.simplefilter(action='ignore', category=FutureWarning)





def main():
    
    mutation_differences_df = pd.read_csv(
        "/home/az2798/MDmis/data/calvados_mutation_differences.csv", index_col=0
    )
    print(mutation_differences_df["Mutation_Category"].value_counts())
    length_palette = {"Pathogenic - Long IDRs": "#e81a1a", "Pathogenic - Short IDRs": "#f5ed11",
                      "Benign": "#3497ed"}
    results_dir = "/home/az2798/MDmis/results/CALVADOS_figures/"
    significance_levels = {0.0001: '***', 0.001: '**', 0.05: '*'}
    
    # plot_boxplot_with_significance(
    #     mutation_differences_df,
    #     "Mutation_Category",
    #     "Res_MD_Diff_1",
    #     results_dir, "SASA_Difference.png",
    #     significance_levels,
    #     title= 'Mutated vs WT MD', xlabel='Length Category', ylabel='SASA Difference',
    #     plot_type = "violin", bar_height = 0.4, palette = length_palette
    # )
    
    # plt.clf()

    plot_ridgeplot(
        data=mutation_differences_df,
        group_col='Mutation_Category',
        value_col="Res_MD_Diff_1",
        results_dir=results_dir,
        xlabel = "SASA Difference",
        plot_filename="SASA_Difference.png",
        label_placement ="left", palette = length_palette, 
        xlim = (-2,2), bar_height = 0.025, xlim_buffer = 1.025
    )
    plt.clf()

    plot_boxplot_with_significance(
        mutation_differences_df,
        "Mutation_Category",
        "Res_MD_Diff_3",
        results_dir, "RMSF_Difference.png",
        significance_levels,
        title='Mutated vs WT MD', xlabel='Length Category', ylabel='RMSF Difference',
        plot_type = "violin", palette = length_palette
    )
    

    plt.clf()

    plot_boxplot_with_significance(
        mutation_differences_df,
        "Mutation_Category",
        "Res_MD_FC_4",
        results_dir, "B_Bridge_FC.png",
        significance_levels,
        title='Mutated vs WT MD', xlabel='Length Category', ylabel='Beta Bridges Ratio',
        plot_type = "violin", palette = length_palette
    )
    
    plt.clf()

    plot_boxplot_with_significance(
        mutation_differences_df,
        "Mutation_Category",
        "Res_MD_Diff_4",
        results_dir, "B_Bridge_Diff.png",
        significance_levels,
        title='Mutated vs WT MD', xlabel='Length Category', ylabel='Beta Bridges Diff',
        plot_type = "violin", palette = length_palette
    )
    
    plt.clf()


    plot_ridgeplot(
        data=mutation_differences_df,
        group_col='Mutation_Category',
        value_col="Res_MD_FC_8",
        results_dir=results_dir,
        xlabel = "3 Helix Ratio",
        plot_filename="3_Helix_FC.png",
        label_placement ="right", palette = length_palette, 
        xlim = (0,4), bar_height = 0.05, xlim_buffer = 1.025
    )
    plt.clf()

    plot_boxplot_with_significance(
        mutation_differences_df,
        "Mutation_Category",
        "Res_MD_Diff_8",
        results_dir, "3_Helix_Diff.png",
        significance_levels,
        title='Mutated vs WT MD', xlabel='Length Category', ylabel='3 Helix Diff',
        plot_type = "violin", palette = length_palette
    )
    plt.clf()

    # plot_boxplot_with_significance(
    #     mutation_differences_df,
    #     "Mutation_Category",
    #     "Avg_Pair_MD_Diff_1",
    #     results_dir, "Avg_Pair_1.png",
    #     significance_levels,
    #     title='', xlabel='Length Category', ylabel='Difference in VDW Forces',
    #     plot_type = "violin", bar_height = 2, palette = length_palette, verbose = True
    # )
    # plt.clf()

    plot_ridgeplot(
        data=mutation_differences_df,
        group_col='Mutation_Category',
        value_col="Avg_Pair_MD_Diff_1",
        results_dir=results_dir,
        xlabel = "Difference in VDW Forces",
        plot_filename="VDW_Differences.png",
        label_placement ="left", palette = length_palette, 
        xlim = (-20,0), bar_height = 0.025, xlim_buffer = 1.025
    )
    plt.clf()

    plot_boxplot_with_significance(
        mutation_differences_df,
        "Mutation_Category",
        "Avg_Pair_MD_FC_1",
        results_dir, "Avg_Pair_FC_1.png",
        significance_levels,
        title='', xlabel='Length Category', ylabel='Ratio of VDW Forces',
        plot_type = "violin", bar_height = 2, palette = length_palette
    )
    plt.clf()

    plot_ridgeplot(
        data=mutation_differences_df,
        group_col='Mutation_Category',
        value_col="Avg_Pair_MD_Diff_2",
        results_dir=results_dir,
        xlabel = "Difference in Hydrogen Bonds\nBB-BB",
        plot_filename="HBBB_Diff.png",
        label_placement ="right", palette = length_palette,
        bar_height = 0.025, xlim_buffer = 1.025, type="bar"
    )
    plt.clf()

    plot_boxplot_with_significance(
        mutation_differences_df,
        "Mutation_Category",
        "Avg_Pair_MD_Diff_3",
        results_dir, "Avg_Pair_3.png",
        significance_levels,
        title='', xlabel='Length Category', ylabel='Difference in HBSB',
        plot_type = "violin",  bar_height = 2, palette = length_palette
    )
    plt.clf()

    plot_boxplot_with_significance(
        mutation_differences_df,
        "Mutation_Category",
        "Avg_Pair_MD_Diff_10",
        results_dir, "Avg_Pair_10.png",
        significance_levels,
        title='', xlabel='Length Category', ylabel='Difference in Covariance',
        plot_type = "violin",  bar_height = 0.05, palette = length_palette
    )
    plt.clf()
    
    # plot_boxplot_with_significance(
    #     mutation_differences_df,
    #     "Mutation_Category",
    #     "Res_All_MD_Diff_1",
    #     results_dir, "Res_All_MD_Diff_1.png",
    #     significance_levels,
    #     title='', xlabel='Length Category', ylabel='Average Difference in SASA',
    #     plot_type = "violin", palette = length_palette
    # )
    # plt.clf()

    # plot_boxplot_with_significance(
    #     mutation_differences_df,
    #     "Mutation_Category",
    #     "Res_All_MD_Diff_19",
    #     results_dir, "Res_All_MD_Diff_19.png",
    #     significance_levels,
    #     title='', xlabel='Length Category', ylabel='Average Difference in Chi Angle (8th)',
    #     plot_type = "violin", bar_height = 0.05, palette = length_palette
    # )
    # plt.clf()

    # plot_boxplot_with_significance(
    #     mutation_differences_df,
    #     "Mutation_Category",
    #     "Res_All_MD_Diff_35",
    #     results_dir, "Res_All_MD_Diff_35.png",
    #     significance_levels,
    #     title='', xlabel='Length Category', ylabel='Average Difference in Phi Angle (12th)',
    #     plot_type = "violin", bar_height = 0.01, palette = length_palette
    # )
    # plt.clf()

    # plot_boxplot_with_significance(
    #     mutation_differences_df,
    #     "Mutation_Category",
    #     "Res_Euclid",
    #     results_dir, "Res_Euclid.png",
    #     significance_levels,
    #     title='', xlabel='Length Category', ylabel='Residue Euclidean Distance',
    #     plot_type = "violin"
    # )
    # plt.clf()

    plot_boxplot_with_significance(
        mutation_differences_df,
        "Mutation_Category",
        "Rg_Difference",
        results_dir, "Rg_Diff.png",
        significance_levels,
        title='', xlabel='Length Category', ylabel='Difference in Radius of Gyration',
        plot_type = "violin"
    )
    plt.clf()

    plot_boxplot_with_significance(
        mutation_differences_df,
        "Mutation_Category",
        "Ete_Distance",
        results_dir, "Ete_Diff.png",
        significance_levels,
        title='', xlabel='Length Category', ylabel='Difference in ETE Distance',
        plot_type = "violin"
    )
    plt.clf()

    # plot_boxplot_with_significance(
    #     mutation_differences_df,
    #     "Mutation_Category",
    #     "nu_Difference",
    #     results_dir, "nu_Diff.png",
    #     significance_levels,
    #     title='', xlabel='Length Category', ylabel='Difference in Nu',
    #     plot_type = "violin", palette = length_palette
    # )
    # plt.clf()

    plot_ridgeplot(
        data=mutation_differences_df,
        group_col='Mutation_Category',
        value_col="nu_Difference",
        results_dir=results_dir,
        xlabel = "Difference in Nu",
        plot_filename="nu_Diff.png",
        label_placement ="right", palette = length_palette, 
        xlim = (-0.07, 0.07), bar_height = 0.025, xlim_buffer = 1.025
    )
    plt.clf()

if __name__ == "__main__":
    main()
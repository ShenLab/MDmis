import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.stats as ss
import itertools
from collections import Counter
import scipy.stats as ss
import glob
from Bio import SeqIO
import math
import tqdm
import sys
import pathlib
ROOT = pathlib.Path(__file__).parents[1]
sys.path.append(str(ROOT))
from utils import *
from config import config



def over_slice(str, window_size):
        '''
        Function that takes a string and parses into substrings of window size        
        :param str: Input string
        :param window_size: Window size for the substrings
        '''
        
        itr = iter(str)
        res = tuple(itertools.islice(itr, window_size))
        if len(res) == window_size:
            yield res    
        for ele in itr:
            res = res[1:] + (ele,)
            yield res
def extract_LCR_polyX(s): # for each sub-sequence, check if it is an
                            # LCRs of polyX pattern 
    if s == len(s) * s[0]:
        return 1
    else:
        return 0
def extract_LCR_polyXY(s):
    counter = Counter(s)
    polyXY= True
    if len(counter.keys()) == 2:
        # only two unique AAs
        for aa in counter.keys():
            num_appearances = counter[aa]
            if num_appearances<2:
                polyXY = False
    else:
        polyXY = False
    if polyXY:
        return 1
    else:
        return 0
def compute_seq_composition(IDR_metadata, seq_column_name, LCR_polyX_window_size,
                            LCR_polyXY_window_size):
    """
    Parameters
    ----------
    IDR_metadata: pandas.DataFrame
    Dataframe of IDRs with the start to end information and the wild-type sequences

    seq_column_name: str
    Column name of the dataframe with the wild-type sequences

    LCR_polyX_window_size: int
    An integer for the number of consecutive residues that constitute a "repeat" -- for determining low-complexity regions of
     type polyX (single amino-acid) 

    LCR_polyXY_window_size: int
    An integer for the number of consecutive residues that constitute a "repeat" -- for determining low-complexity regions of
     type polyXY (two amino-acids occurring in a pattern) 

    """
    
    polyX_column = []
    polyXY_column =[]
    for i, row in IDR_metadata.iterrows():
        sequence = row[seq_column_name]
        X_subseqs = ["".join(ele) for ele in over_slice(sequence, LCR_polyX_window_size)]
        XY_subseqs = ["".join(ele) for ele in over_slice(sequence, LCR_polyXY_window_size)]
        
        polyX_occurrences = np.sum(list(map(extract_LCR_polyX, X_subseqs))) / \
            (len(sequence)-LCR_polyX_window_size + 1)
        polyXY_occurrences = np.sum(list(map(extract_LCR_polyXY, XY_subseqs))) / \
            (len(sequence)-LCR_polyXY_window_size + 1)

        polyX_column.append(polyX_occurrences)
        polyXY_column.append(polyXY_occurrences)

    return polyX_column, polyXY_column

def test():
    str1 = "ARARARGGGGGG" #1 of each
    str2 = "ARGNARGNARGN" #neither

    str1_substrings_X = ["".join(ele) for ele in over_slice(str1, 6)]
    str1_substrings_XY = ["".join(ele) for ele in over_slice(str1, 6)]
    polyX_occurrences = np.sum(list(map(extract_LCR_polyX, str1_substrings_X))) / \
            (len(str1)-6 + 1)
    assert polyX_occurrences == 1/7

    polyXY_occurrences = np.sum(list(map(extract_LCR_polyXY, str1_substrings_XY))) / \
            (len(str1)-6 + 1)
    assert polyXY_occurrences == 1/7

def main():
    data_dir = os.path.abspath(config["data_dir"])
    results_dir = os.path.abspath(config["results_dir"])
    MSA_dir = os.path.abspath(config["MSA_dir"])
    aa_order = "ARNDCQEGHILKMFPSTWYV"

    MD_metadata = pd.read_csv(os.path.join(data_dir, "MD_metadata.csv"), index_col=0)
    
    MD_metadata.rename(columns={"source": "MD Data Source"}, inplace=True)
    IDRome_metadata = MD_metadata[MD_metadata["MD Data Source"] == "IDRome"]
    print(IDRome_metadata.head())
    test()  

    polyX_column, polyXY_column = compute_seq_composition(IDRome_metadata, "sequence", 6, 10)

    IDRome_metadata["polyX_LCR"] = polyX_column
    IDRome_metadata["polyXY_LCR"] = polyXY_column
    IDRome_metadata["Length"] = IDRome_metadata["sequence"].apply(len)

    print(IDRome_metadata.head())

    ### Plot out the results
    
    plt.figure(figsize=(6,6))
    r, p = ss.spearmanr(IDRome_metadata["polyX_LCR"], IDRome_metadata["Length"])
    ax = sns.regplot(data=IDRome_metadata, x= "Length", y= "polyX_LCR")
    plt.xlabel("IDR Length (residues)", fontdict={"size":16})
    plt.xscale("log")
    plt.ylabel("Proportion of Poly X Patterns",fontdict={"size":16})

    plt.title("Monomeric Repeats and Length",fontdict={"size":18})
    plt.text(0.45, 0.9, f"ρ: {r:.2f}, p-value: {p:.2f}", fontdict={"size":16},transform = ax.transAxes)
    plt.savefig(os.path.join(config["results_dir"], "clinical_figures", "polyX_Length.png"), dpi = 200, bbox_inches = "tight")

    plt.close()

    plt.figure(figsize=(6,6))
    r, p = ss.spearmanr(IDRome_metadata["polyXY_LCR"], IDRome_metadata["Length"])
    ax = sns.regplot(data=IDRome_metadata, x= "Length", y= "polyXY_LCR")
    plt.xlabel("IDR Length (residues)", fontdict={"size":16})
    plt.xscale("log")

    plt.ylabel("Proportion of Poly XY Patterns",fontdict={"size":16})
    plt.title("Dimeric Repeats and Length",fontdict={"size":18})
    plt.text(0.45, 0.9, f"ρ: {r:.2f}, p-value: {p:.2f}", fontdict={"size":16}, transform = ax.transAxes)
    plt.savefig(os.path.join(config["results_dir"], "clinical_figures", "polyXY_Length.png"), dpi = 200, bbox_inches = "tight")
    plt.close()

    IDRome_metadata['Length Category'] = np.select(
        [IDRome_metadata["Length"] <= 40,
        (IDRome_metadata["Length"] > 40) & (IDRome_metadata["Length"]<=100),
        (IDRome_metadata["Length"] > 100) & (IDRome_metadata["Length"]<=200),
        (IDRome_metadata["Length"] > 200) & (IDRome_metadata["Length"]<=800),
        IDRome_metadata["Length"] > 800
        ],
        ['<=40aa', "40-100aa", "100-200aa", "200-800aa", '>800aa']
    )

    plt.figure(figsize=(6,6))
    stat, p = ss.mannwhitneyu(IDRome_metadata[IDRome_metadata["Length Category"]=="<=40aa"]["polyX_LCR"], 
                                IDRome_metadata[IDRome_metadata["Length Category"]=="40-100aa"]["polyX_LCR"])
    ax = sns.barplot(data=IDRome_metadata, x= "Length Category", y= "polyX_LCR")
    plt.xlabel("Length", fontdict={"size":16})
    plt.ylabel("Proportion of Poly X Patterns",fontdict={"size":16})
    plt.title("Monomeric Repeats and Length",fontdict={"size":18})
    #plt.text(0.45, 0.9, f"U: {r:.2f}, p-value: {p:.2f}", fontdict={"size":16}, transform = ax.transAxes)
    plt.savefig(os.path.join(config["results_dir"], "clinical_figures", "polyX_Length_boxplot.png"), dpi = 200, bbox_inches = "tight")
    plt.close()

    plt.figure(figsize=(6,6))
    stat, p = ss.mannwhitneyu(IDRome_metadata[IDRome_metadata["Length Category"]=="<=40aa"]["polyXY_LCR"], 
                                IDRome_metadata[IDRome_metadata["Length Category"]=="40-100aa"]["polyXY_LCR"])
    ax = sns.barplot(data=IDRome_metadata, x= "Length Category", y= "polyXY_LCR")
    plt.xlabel("Length", fontdict={"size":16})
    plt.ylabel("Proportion of Poly XY Patterns",fontdict={"size":16})
    plt.title("Dimeric Repeats and Length",fontdict={"size":18})
    #plt.text(0.45, 0.9, f"U: {r:.2f}, p-value: {p:.2f}", fontdict={"size":16}, transform = ax.transAxes)
    plt.savefig(os.path.join(config["results_dir"], "clinical_figures", "polyXY_Length_boxplot.png"), dpi = 200, bbox_inches = "tight")
    plt.close()

if __name__ == "__main__":
    main()


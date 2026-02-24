import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
import scipy.stats as ss
import numpy as np
import os
import re
import warnings
import time
import argparse
from CALVADOS_code.CALVADOS_utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.rcParams.update({'font.size': 13})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--directory', 
                        help = "Indicates the target directory for the simulations.")
    args = parser.parse_args()
    
    data_dir = "/home/az2798/MDmis/data/"
    
    results_dir = args.directory
    residues_file = os.path.join(data_dir, "residues.csv")
    #proteins_to_simulate = ["P02458_78_1254"]
    proteins_to_simulate = ["O75179_1794_2603",
                            "P08123_1_1134",
                            "P29400_1_1456",
                            "Q01955_1_1441",
                            "Q12955_1434_2560",
                            "Q9UM47_1_1399",
                            "A2RUB1_1_749",
                            "Q6KC79_92_1196",
                            "P46100_289_1546"]

    #starting with top 50
    MD_metadata = pd.read_csv(os.path.join(data_dir, "MD_metadata.csv"),
                            index_col= 0 )

    IDRome_metadata = MD_metadata[MD_metadata["source"]== "IDRome"]


    #running experiment where we split the protein in two halves and simulate each half separately

    for protein_region in proteins_to_simulate:
        
        uniprot_id, start, end = protein_region.split("_")
        start, end = int(start), int(end)

        sequence = IDRome_metadata.loc[IDRome_metadata["protein_start_end"] == protein_region]["sequence"].values[0]
        midpoint = len(sequence)//2
        sequence_half1, sequence_half2 = sequence[0:midpoint], sequence[midpoint:] #split into two halves
        #Simulating the first half
        time_start = time.time()
        protein_region_half1 = f"{uniprot_id}_{start}_{start+midpoint-1}" #NAME to be stored 
    
        print(f"Running {protein_region_half1}.")
        run_md_sim(protein_region_half1,
                sequence_half1,
                residues_file,
                results_dir,
                charged_N_terminal_amine=True,
                charged_C_terminal_carboxyl = False,
                charged_histidine = False,
                Simulation_time = "AUTO") #set simulation time to 150ns for short simulations
        time_end = time.time()
        print(f"Finished {protein_region} in {time_end - time_start} seconds.")

        #Second half
        protein_region_half2 = f"{uniprot_id}_{start+midpoint}_{end}"
        time_start = time.time()
        print(f"Running {protein_region_half2}.")
        run_md_sim(protein_region_half2, 
                sequence_half2,
                residues_file,
                results_dir,
                charged_N_terminal_amine=True,
                charged_C_terminal_carboxyl = False,
                charged_histidine = False,
                Simulation_time = "AUTO") #set simulation time to 150ns for short simulations
        time_end = time.time()

if __name__ == "__main__":
    main()

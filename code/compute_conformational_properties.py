import subprocess
import os
import re
import warnings
import time
import argparse
import shutil
import glob
from process_md_trajectory import *
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd



    
def main():
    data_dir = "/home/az2798/MDmis/data/"
    residues_masses = pd.read_csv(os.path.join(
        data_dir, "residues_masses.csv"
    ), index_col=0)
    
    output_directory = "/nfs/user/Users/az2798/processed_CALVADOS/Benign/"
    CALVADOS_runs_directory = "/nfs/user/Users/az2798/CALVADOS_runs/Benign/"
    for folder in glob.glob(os.path.join(CALVADOS_runs_directory, "*")):
        mutation_name = os.path.basename(folder)
        print(mutation_name)
        if mutation_name == "Q86UQ4_2597_3287_F:2799:L":
            continue
            
        compute_conformational_properties(
                    os.path.join(CALVADOS_runs_directory, mutation_name, 'traj.dcd'),
                    os.path.join(CALVADOS_runs_directory, mutation_name, 'top.pdb'),
                    residues_masses,
                    os.path.join(output_directory, mutation_name, 'conformational_properties.npy'),
                     0.02, 800)
        

    output_directory = "/nfs/user/Users/az2798/processed_CALVADOS/Pathogenic_High_RMSF/"
    CALVADOS_runs_directory = "/nfs/user/Users/az2798/CALVADOS_runs/Pathogenic_High_RMSF_ld2/"
    for folder in glob.glob(os.path.join(CALVADOS_runs_directory, "*")):
        mutation_name = os.path.basename(folder)
        print(mutation_name)

        compute_conformational_properties(
                    os.path.join(CALVADOS_runs_directory, mutation_name, 'traj.dcd'),
                    os.path.join(CALVADOS_runs_directory, mutation_name, 'top.pdb'),
                    residues_masses,
                    os.path.join(output_directory, mutation_name, 'conformational_properties.npy'),
                     0.02, 800)
    

    output_directory = "/nfs/user/Users/az2798/processed_CALVADOS/Pathogenic_Low_RMSF/"
    CALVADOS_runs_directory = "/nfs/user/Users/az2798/CALVADOS_runs/Pathogenic_Low_RMSF/"
    for folder in glob.glob(os.path.join(CALVADOS_runs_directory, "*")):
        mutation_name = os.path.basename(folder)
        print(mutation_name)

        compute_conformational_properties(
                    os.path.join(CALVADOS_runs_directory, mutation_name, 'traj.dcd'), ##use coarse-grained simulation for replication
                    os.path.join(CALVADOS_runs_directory, mutation_name, 'top.pdb'),
                    residues_masses,
                    os.path.join(output_directory, mutation_name, 'conformational_properties.npy'),
                     0.02, 800)

    # output_directory = "/nfs/user/Users/az2798/processed_CALVADOS/Shorter_simulations/"
    # CALVADOS_runs_directory = "/nfs/user/Users/az2798/CALVADOS_runs/Shorter_simulations/"
    # for folder in glob.glob(os.path.join(CALVADOS_runs_directory, "*")):
    #     mutation_name = os.path.basename(folder)
    #     print(mutation_name)

    #     compute_conformational_properties(
    #                 os.path.join(CALVADOS_runs_directory, mutation_name, 'traj.dcd'), ##use coarse-grained simulation for replication
    #                 os.path.join(CALVADOS_runs_directory, mutation_name, 'top.pdb'),
    #                 residues_masses,
    #                 os.path.join(output_directory, mutation_name, 'conformational_properties.npy'),
    #                  0.02, 10)
if __name__ == "__main__":
    main()
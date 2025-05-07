import subprocess
import os
import warnings
import pathlib
import glob
from process_md_trajectory import *
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

ROOT = pathlib.Path(__file__).parent
sys.path.append(ROOT)
from config import config


    
def main():
    data_dir = os.path.abspath(config["data_dir"])
    residues_masses = pd.read_csv(os.path.join(
        data_dir, "residues_masses.csv" #obtained from Tesei 2024's GitHub (residues.csv)
    ), index_col=0)
        

    output_directory = os.path.abspath(config["CALVADOS_raw_dir"]) #note that this needs to be run for each subtype (Pathogenic High RMSF, Low RMSF, Benign separately.)
    CALVADOS_runs_directory = os.path.abspath(config["CALVADOS_processed_dir"])
    for folder in glob.glob(os.path.join(CALVADOS_runs_directory, "*")):
        mutation_name = os.path.basename(folder)
        print(mutation_name)
        if mutation_name == "Q86UQ4_2597_3287_F:2799:L": #Skip for Benign!
            continue
        compute_conformational_properties(
                    os.path.join(CALVADOS_runs_directory, mutation_name, 'traj.dcd'),
                    os.path.join(CALVADOS_runs_directory, mutation_name, 'top.pdb'),
                    residues_masses,
                    os.path.join(output_directory, mutation_name, 'conformational_properties.npy'),
                     0.02, 800)
    

    
if __name__ == "__main__":
    main()
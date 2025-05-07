import subprocess
import os
import warnings

import shutil
import glob
import pandas as pd
import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.append(ROOT)
from utils import *
from config import config
from process_md_trajectory import *
warnings.simplefilter(action='ignore', category=FutureWarning)



def convert_cg_to_aa(CG_topology_filename, CG_trajectory_filename,
                     mutation_name,
                     cg2all_script,
                     getcontacts_script,
                     output_directory):
    
    if os.path.exists(os.path.join(output_directory, mutation_name, "res_feature.npy")):
        print(f"{mutation_name} is already processed. Skipping it.")
        return True
    elif os.path.exists(os.path.join(output_directory, mutation_name)):
        print(f"{mutation_name} has a folder but no processed features. Deleting, and starting fresh.")
        shutil.rmtree(os.path.join(output_directory, mutation_name))
    
    os.mkdir(os.path.join(output_directory, mutation_name))

    subprocess.run(["bash", cg2all_script, 
                    CG_topology_filename, CG_trajectory_filename,
                    os.path.join(output_directory, mutation_name,
                                 "aa_traj.dcd"),
                    os.path.join(output_directory, mutation_name,
                                 "aa_top.pdb")])
    
    subprocess.run(["bash", getcontacts_script,
                    os.path.join(output_directory, mutation_name,
                                 "aa_top.pdb"),
                    os.path.join(output_directory, mutation_name,
                                 "aa_traj.dcd"),
                    os.path.join(output_directory, mutation_name,
                                 "contacts.tsv")])
    
    return False
    
def main():
    data_dir = os.path.abspath(config["data_dir"])
    residues_masses = pd.read_csv(os.path.join(
        data_dir, "residues_masses.csv"
    ), index_col=0)
    cg2all_script = "run_cg2all.sh" #should be in same folder
    getcontacts_script = "run_GetContacts.sh"
    
    
    output_directory = os.path.abspath(config["CALVADOS_processed_dir"]) #This needs to be specific to each mutation type
    CALVADOS_runs_directory = os.path.abspath(config["CALVADOS_raw_dir"])
    
  
    for folder in glob.glob(os.path.join(CALVADOS_runs_directory, "*")):
        mutation_name = os.path.basename(folder)
        print(mutation_name)
        if mutation_name == "Q86UQ4_2597_3287_F:2799:L": #Skip for Benign!
            continue
        already_processed = convert_cg_to_aa(os.path.join(folder, "top.pdb"),
                         os.path.join(folder, "traj.dcd"),
                         mutation_name, 
                         cg2all_script,
                         getcontacts_script,
                         output_directory)
        if not already_processed:
            process_md_data(os.path.join(output_directory, mutation_name, 'aa_traj.dcd'),
                        os.path.join(output_directory, mutation_name, 'aa_top.pdb'),
                        os.path.join(output_directory, mutation_name, 'contacts.tsv'),
                        os.path.join(output_directory, mutation_name, 'res_feature.npy'),
                        os.path.join(output_directory, mutation_name, 'pair_feature.npy'),
                            0.02, 800)
            compute_conformational_properties(
                        os.path.join(CALVADOS_runs_directory, mutation_name, 'traj.dcd'), ##use coarse-grained simulation for replication
                        os.path.join(CALVADOS_runs_directory, mutation_name, 'top.pdb'),
                        residues_masses,
                        os.path.join(output_directory, mutation_name, 'conformational_properties.npy'),
                            0.02, 800)



if __name__ == "__main__":
    main()
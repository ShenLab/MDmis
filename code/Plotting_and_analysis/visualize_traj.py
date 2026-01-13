import py3Dmol
import re
import os
import numpy as np
import pandas as pd
import scipy.stats as ss

IDRome_cg_dir = "/nfs/user/Users/ch3849/descartes/prodance/disorder/IDRome/IDRome_v4"
IDRome_aa_dir = "/nfs/user/Users/ch3849/descartes/prodance/disorder/IDRome/cg2all" 

# WT CG structure

pdb_path = f"{IDRome_cg_dir}/P2/94/00/1_1456/top.pdb" #COL4A5

# create view
view = py3Dmol.view(width=500, height=500)
view.addModel(pdb_path, "pdb")

view.setStyle({"cartoon": {"color": "spectrum"}})
view.zoomTo()

# 
pdb_path = f"{IDRome_aa_dir}/00/P29400_1_1456.pdb" #COL4A5

# create view
view = py3Dmol.view(width=500, height=500)
view.addModel(pdb_path, "pdb")

view.setStyle({"cartoon": {"color": "spectrum"}})
view.zoomTo()
 
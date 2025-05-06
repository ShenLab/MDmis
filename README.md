# MDmis - A biophysical machine learning approach for missense variants in disordered protein regions

In this project, we leverage molecular dynamics (MD) simulations of long time scales of intrinsicially disordered regions of human proteome. We try to understand how AlphaMissense and ESM1b predict pathogenic variants in IDRs, and use MDmis to interpret predictions in addition to identify subtypes of pathogenic variation. 
We perform an analysis of pathogenic variants in IDRs using features from MD, in tandem with
functional annotations of protein residue sites and protein regions, and we perform simulations of mutated primary sequences to verify and supplement our hypotheses from Wild-type simulations.

# Code and general information
This repository contains the code used to train MDmis, perform analysis, run simulations of mutated sequences, and extract features. All code uses Python and Command Line.

Majority of the training and analysis can be done using MDmis_env.yml. You can install this envionment as:
```conda env create -f MDmis_env.yml``` 

Other analysis, such as simulating MD trajectories using CALVADOS2, converting coarse-grained trajectories to all-atom, and then processing MD trajectories with GetContacts require their own environments due to conflicts in packages and their versions. We refer you to 
1. https://github.com/KULL-Centre/CALVADOS.git
2. https://github.com/huhlim/cg2all.git
3. https://github.com/getcontacts/getcontacts.git
   
Respectively for their codebase and setup. 

# Data
All processed data used in these analyses are provided in our Zenodo repository. The DOI for our repository is: 10.5281/zenodo.15346250

# Contact
For any specific information about code or data processing, you may reach out az2798@cumc.columbia.edu or publish an issue on our GitHub repo.

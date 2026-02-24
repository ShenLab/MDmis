# MDmis - A biophysical machine learning approach for missense variants in disordered protein regions

In this project, we leverage molecular dynamics (MD) simulations of long time scales of intrinsicially disordered regions of human proteome. We try to understand how AlphaMissense and ESM1b predict pathogenic variants in IDRs, and use MDmis to interpret predictions in addition to identify subtypes of pathogenic variation. 
We perform an analysis of pathogenic variants in IDRs using features from MD, in tandem with
functional annotations of protein residue sites and protein regions, and we perform simulations of mutated primary sequences to verify and supplement our hypotheses from Wild-type simulations.
![Complete_MDmis_Workflow (1)](https://github.com/user-attachments/assets/5746a846-454d-4531-9f06-0792a1c11958)

# Code and general information
This repository contains the code used to train MDmis, perform analysis, run simulations of mutated sequences, and extract features. All code uses Python and Command Line.

## Installation
We recommend installing the prequisite packages using pip on conda as listed below:

. pandas

. numpy

. scipy

. scikit-learn

. matplotlib

. seaborn

. joblib

. pickle

. tqdm

. biopython

You may also opt to use our environment file MDmis_env.yml. You can install this envionment as:


```conda env create -f MDmis_env.yml``` 
**Note that it may only work for linux OS.**

## Other environments needed for additional analyses
Other analysis, such as simulating MD trajectories using CALVADOS2, converting coarse-grained trajectories to all-atom, and then processing MD trajectories with GetContacts require their own environments due to conflicts in packages and their versions. We refer you to 
1. https://github.com/KULL-Centre/CALVADOS.git
2. https://github.com/huhlim/cg2all.git
3. https://github.com/getcontacts/getcontacts.git
   
Respectively for their codebase and setup. 

## Example code usage (for training MDmis) and output
1. Edit ```config.py``` with correct paths based on your project setup.
   
2. ```python code/Data_processing/create_feature_clinical_table.py --interaction_cutoff 0.4 --window_size 3```

Example output: {data_dir}/clinical_train_val_interactioncutoff_0.4/feature_table.csv

3. ```python code/Model/train_MDmis_RF.py --interaction_cutoff 0.4```
 
Example outputs: 

{data_dir}/clinical_train_val_interactioncutoff_0.4/fold_{k}/train.csv and {data_dir}/clinical_train_val_interactioncutoff_0.4/fold_{k}/val.csv

{model_dir}/clinical_train_val_interactioncutoff_0.4/fold_{k}/{model_prefix}.pkl

4. ```python code/Model/evaluate_MDmis.py --interaction_cutoff 0.4```
5. 
Example outputs: {results_dir}/Clinical_ROCs/{interaction_cutoff}/{testing_set}.png

6. If you want to create your own IDRome-wide processed data set with model predictions

```python code/Model/create_feature_table_entire_IDRome.py --interaction_cutoff 0.4 --chunk 0```

Example output: {vault_dir}/MDmis_predictions/{IDRome_ID}/predictions.csv

# Data
All processed data used in these analyses are provided in our Zenodo repository. The DOI for our repository is: 10.5281/zenodo.15346250

# Contact
For any specific information about code or data processing, you may reach out az2798@cumc.columbia.edu or publish an issue on our GitHub repo.

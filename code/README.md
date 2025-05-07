The code and scripts used in "Molecular dynamics simulations of intrinsically disordered protein regions enable biophysical interpretation of variant effect predictors". 
For reproducing the study's results and figures, you may follow the steps outlined below.

## 0. Data pre-processing
1. The data pre-processing folder allows processing of DMS labels by extracting labels from ProteinGym files of assays. It requires a DMS_metadata file that can be found on our Zenodo.
2. You may also process the dbNSFP database to extract GERP RS scores, and other minor tasks. dbNSFP can be found at this [link](https://www.dbnsfp.org/)
3. You may also need to process the LLR scores from ESM1b that is given for each isoform separately.

## 1. Data processing
Prequisites: training.csv file from Zenodo, downloaded AlphaMissense scores, processed ESM1b LLR, pLDDT from AlphaFold, Zoonomia MSAs converted to protein sequences, and .h5py file containing dynamic features from our [HuggingFace](https://huggingface.co/datasets/ChaoHou/protein_dynamic_properties).

1. Most of the data work is done using process_proteome_information.py which merges labels with conservation information, alphamissense scores, and ESM1b scores.
2. To create the feature tables, use create_feature_table_clinical.py for the Clinvar, PrimateAI, and other labels in IDRs. For Deep mutational scans, use create_feature_table_DMS.py. create_feature_table_GPCRmd.py was used to extract basic dynamics features as a positive control.
3. feature_extraction_MSA.py uses Zoonomia MSAs to extract features such as entropy, compositional bias, and charge pattern.
4. Note that utils.py (in the code root directory) has the function to extract information from the MD simulations into a feature table for training.

## 2. Model training and testing/evaluation
Prerequistes: either download the **feature_table.csv** from our Zenodo or create it using the dynamic features and the steps above. Processed DMS labels and features from pre-processing and steps above.
1. train_MDmis_RF.py is used for the training tasks.
2. evaluate_MDmis.py is used to evaluate MDmis, AlphaMissense, and ESM1b on clinical labels.
3. evaluate_MDmis_DMS.py is used to evaluate these models on DMS labels.
4. predict_MDmis.py has helper functions to faciliate these evaluations.
5. utils.py has the functions needed to plot ROC curves for the clinical evaluation and return spearman Rho values for the DMS evaluation

## 3. Plotting and Analysis
Prerequisites: Processed proteome information dataframe can be found on our Zenodo or created as above. Feature tables (found or created above) are also needed since they contain dynamic features in tabular format. The entire database of dbPTM with all of the modification types, the data file from PhaSepDB, and the data file from TFRegDB will be needed from the respective databases.
1. The Plotting and Analysis folder contains code needed to perform analysis of MD features and variant types. Additionally, there are scripts to analyze IDRs and their annotations (such as PTMs, TFregDB, Phase Separation etc.).
2. utils.py and plotter_functions.py contain many helper functions to faciliate plotting of different figures and relationships.
3. plot_variant_features.py generates majority of the figures for the manuscript.

## 4. Simulations of mutated sequences and plotting/analysis
Prerequisites: The feature table (preferably split into folds and merged with MSA features) and the CALVADOS GitHub [repository](https://github.com/KULL-Centre/CALVADOS.git) cloned.

1. Running MD simulations using CALVADOS2 was done using code from their repository. It is provided in the folder titled CALVADOS_code.
2. To convert coarse-grained MD simulations to all-atom trajectories and the MD features, you may use convert_CALVADOS_AA.py which performs both tasks.
  a. It does so by using convert_cg2all.sh and run_GetContacts.sh
  b. It also performs process_md_trajectory.py and compute_conformational_properties.py
**An important note about convert_cg2all.sh and run_GetContacts.sh. The directories in th bash scripts need to be added based on your conda environments (if using) and the GetContacts cloned GitHub [repository](https://github.com/getcontacts/getcontacts.git)**
3. Using this code by invoking single missense variants is done by run_CALVADOS_mutations_batches.py, which takes arguments to run batches of mutations.
4. To compute differences between single missense variants and their wild-type MD simulations, use analyze_CALVADOS_mutations.py, and then plot_CALVADOS_differences from the Plotting and Analysis folder can help visualize differences.


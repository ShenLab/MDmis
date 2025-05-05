The code and scripts used in "Molecular dynamics simulations of intrinsically disordered protein regions enable biophysical interpretation of variant effect predictors". 
For reproducing the study's results and figures, you may follow the steps outlined below.

## Data pre-processing
1. The data pre-processing folder allows processing of DMS labels, the dbNSFP database to extract GERP RS scores, and other minor tasks. 
2. Most of the data work is done using process_proteome_information.py which merges labels with conservation information, alphamissense scores, and ESM1b scores.
3. To create the feature tables, use create_feature_table_clinical.py for the Clinvar, PrimateAI, and other labels in IDRs. For Deep mutational scans, use create_feature_table_DMS.py. create_feature_table_GPCRmd.py was used to extract basic dynamics features as a positive control.
4. feature_extraction_MSA.py uses Zoonomia MSAs to extract features such as entropy, compositional bias, and charge pattern.
5. utils.py has the function to extract information from the MD simulations into a feature table for training.

## Model training and testing/evaluation
1. train_MDmis_RF.py is used for the training tasks.
2. evaluate_MDmis.py is used to evaluate MDmis, AlphaMissense, and ESM1b on clinical labels.
3. evaluate_MDmis_DMS.py is used to evaluate these models on DMS labels.
4. predict_MDmis.py has helper functions to faciliate these evaluations.
5. utils.py has the functions needed to plot ROC curves for the clinical evaluation and return spearman Rho values for the DMS evaluation

## Plotting and Analysis
1. The Plotting and Analysis folder contains code needed to perform analysis of MD features and variant types. Additionally, there are scripts to analyze IDRs and their annotations (such as PTMs, TFregDB, Phase Separation etc.).
2. utils.py and plotter_functions.py contain many helper functions to faciliate plotting of different figures and relationships.
3. plot_variant_features.py generates majority of the figures for the manuscript.

## Simulations of mutated sequences and plotting/analysis
1. Running MD simulations using CALVADOS2 was done using code from their repository. It is provided in the folder titled CALVADOS_code.
2. To convert coarse-grained MD simulations to all-atom trajectories and the MD features, you may use convert_CALVADOS_AA.py which performs both tasks.
  a. It does so by using convert_cg2all.sh and run_GetContacts.sh
  b. It also performs process_md_trajectory.py and compute_conformational_properties.py
3. Using this code by invoking single missense variants is done by run_CALVADOS_mutations_batches.py, which takes arguments to run batches of mutations.
4. To compute differences between single missense variants and their wild-type MD simulations, use analyze_CALVADOS_mutations.py, and then plot_CALVADOS_differences from the Plotting and Analysis folder can help visualize differences.


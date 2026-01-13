config = {
    "code_dir": "/home/az2798/MDmis_revisions/MDmis/code/", #used as the base code folder that has all the other folders
    "vault_dir": "/nfs/user/Users/az2798/", #used for most raw data files which are large (can be the same as data_dir)
    "colabfold_dir": "/nfs/user/Users/dl3738/database/",
    "data_dir": "/home/az2798/MDmis/data/", #used for reading processed data and storing intermediate files
    "results_dir": "/home/az2798/MDmis/results/", #used for storing figures
    "models_dir": "/home/az2798/MDmis/models", #used for saving the .pkl files for the trained RF models
    "CALVADOS_raw_dir": "/nfs/user/Users/az2798/CALVADOS_runs/", #used for storing generated simulations, specific to mutation type (Pathogenic_High_RMSF, Benign etc.)
    "CALVADOS_processed_dir": "/nfs/user/Users/az2798/processed_CALVADOS/", #used for storing processed All-atom simulations and features
    "h5py_path": "/nfs/user/Users/az2798/train_data_all/filtered_feature_all_ATLAS_GPCRmd_IDRome.h5" #contains the tensors of dynamic features of simulations
}
config = {
    "code_dir": "/path/to/dir", #used as the base code folder that has all the other folders
    "vault_dir":  "/path/to/dir", #used for most raw data files which are large (can be the same as data_dir)
    "data_dir":  "/path/to/dir", #used for reading processed data and storing intermediate files
    "results_dir":  "/path/to/dir", #used for storing figures
    "models_dir":  "/path/to/dir", #used for saving the .pkl files for the trained RF models
    "CALVADOS_raw_dir":  "/path/to/dir", #used for storing generated simulations, specific to mutation type (Pathogenic_High_RMSF, Benign etc.)
    "CALVADOS_processed_dir":  "/path/to/dir", #used for storing processed All-atom simulations and features
    "h5py_path":  "/path/to/dir", #contains the tensors of dynamic features of simulations
    "morfchibi_dir": "/path/to/dir" #contains the generated files with MoRF Chibi scores
}
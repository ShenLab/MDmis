import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import math
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
import cProfile
import io 
import pstats
import psutil

from MDmis_modules import *
from train_MDmis import * 

###Date: 06/21/24
###Purpose: To evaluate trained MDmis model. MDmis predicts missense variant pathogenicity
###using features derived from molecular dynamics simulations and AAindex (physiochemical features).

## Defining paths
data_dir = "/share/vault/Users/az2798/"
protein_seq_mapping_dir = "/home/az2798/IDR_cons/data/"
models_dir = "/home/az2798/IDR_cons/intermediate_models/"
results_dir = "/home/az2798/IDR_cons/results/"

def evaluate(trained_model, test_loader, criterion, path_to_save, evaluation_data,
        device='cuda'):

    trained_model.eval()
    all_probabilities = []
    all_variant_matrices = []

    with torch.no_grad():
        for batch in test_loader:
            
            probabilities, variant_matrices = trained_model(batch)
            
            labeled_indices = (variant_matrices == 0) | (variant_matrices == 1)

            all_probabilities.append(probabilities[labeled_indices].cpu())
            all_variant_matrices.append(variant_matrices[labeled_indices].cpu())

    all_probabilities = torch.cat(all_probabilities).numpy()
    all_variant_matrices = torch.cat(all_variant_matrices).numpy()

    # Compute loss
    loss = criterion(torch.tensor(all_probabilities), torch.tensor(all_variant_matrices))
    print(f'Evaluation Loss: {loss.item():.4f}')

    # Flatten arrays for AUROC and AUPRC calculation
    y_true = all_variant_matrices.flatten()
    y_scores = all_probabilities.flatten()

    # Calculate AUROC
    auroc = roc_auc_score(y_true, y_scores)
    print(f'AUROC: {auroc:.4f}')

    # Calculate AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    print(f'AUPRC: {auprc:.4f}')

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for {evaluation_data}, BCELoss: {loss.item():.3f}')
    plt.legend(loc="lower right")
    plt.savefig(f'{path_to_save}{evaluation_data}_roc_curve.png', dpi=300, bbox_inches = "tight")
    # Plot Precision-Recall Curve
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (area = {auprc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall for {evaluation_data}, BCELoss: {loss.item():.3f}')
    plt.legend(loc="lower left")
    plt.savefig(f'{path_to_save}{evaluation_data}_pr_curve.png', dpi=300, bbox_inches = "tight")

    return loss.item(), auroc, auprc    



def main():
    #define device
    gpu_id = 2
    gpu = f'cuda:{gpu_id}'
    device = torch.device(gpu)
    torch.cuda.empty_cache()

    criterion = MisLoss([0.3, 0.7])

    #Data
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    max_seq_len = 128
    stride = 64
    batch_size = 30
    mapped_proteins = pd.read_csv("/home/az2798/IDR_cons/data/mapped_protein_seqs.csv", 
                                  index_col=0)
    mapped_proteins.rename(columns = {"sstart": "start", "send": "end", "UniProtID": "protein_id"}, inplace=True)
    mapped_proteins = mapped_proteins[mapped_proteins["source"]== "IDRome"] #FOR TESTING, 
    
    mapped_proteins["start"] = mapped_proteins["name"].str.split("_").str[1]
    mapped_proteins["end"] = mapped_proteins["name"].str.split("_").str[2]
    
    print(mapped_proteins.head())

    mapped_proteins = mapped_proteins[["sequence", "start", "end", "protein_id", "protein_start_end"]]
    mapped_proteins["start"] = mapped_proteins["start"].astype(int)
    mapped_proteins["end"] = mapped_proteins["end"].astype(int)

    aaindex_res_mat = np.load("/home/az2798/IDR_cons/data/aa_index1.npy")
    aaindex_pair_mat = np.load("/home/az2798/IDR_cons/data/aa_index3.npy")

    clinvar_df = pd.read_csv("/share/vault/Users/az2798/ClinVar_Data/training.csv", index_col=0)
    clinvar_df.rename(columns = {"score":"outcome", "uniprotID":"protein_id", "pos.orig": "location", 
                       "alt": "changed_residue"}, inplace=True)
    clinvar_df = clinvar_df[["outcome", "protein_id", "location", "changed_residue", "VarID"]]
    
    # Subsetting by proteins for which variants are available
    merged_df = pd.merge(left=mapped_proteins, right=clinvar_df, on="protein_id", how="inner")
    merged_df.drop_duplicates(subset="VarID",inplace=True)
    variants_subset_df = merged_df[(merged_df["location"] >= merged_df["start"]) & (merged_df["location"] <= merged_df["end"])]
    proteins_to_subset = set(variants_subset_df["protein_start_end"])

    subset_mapped_proteins = mapped_proteins[mapped_proteins["protein_start_end"].isin(proteins_to_subset)]
    print(subset_mapped_proteins.shape, "Subset Mapped Proteins")
    print(len(subset_mapped_proteins["protein_start_end"].unique()), "Unique protein regions")
    
    print_memory_usage("After loading dataframe")

    #subset_mapped_proteins = subset_mapped_proteins.head(1000)

    h5py_path = "/share/vault/Users/az2798/train_data_all/filtered_feature_all_ATLAS_GPCRmd_IDRome.h5"

    
    res_data, pair_data = load_h5py_data(h5py_path)
    print_memory_usage("After loading h5py data")
    
    train_proteins, val_proteins = train_test_split(subset_mapped_proteins, test_size=0.1, random_state=42) #only for IDRome

    test_protein_dataset = ProteinDataset(val_proteins, aaindex_res_mat, aaindex_pair_mat, clinvar_df,
                                     res_data, pair_data, max_seq_len, stride, aa_order)

    test_loader =  DataLoader(test_protein_dataset, batch_size = batch_size
                             ) 
    
                          

    print(f"Number of data points in test_loader: {len(test_protein_dataset)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")
    

    #define dimensions
    nheads = 8
    
    
    d_in = 47 # residue level MD features
    L = max_seq_len #seqlen
    d_hid = max_seq_len
    d_qkv = d_hid #these should be equivalent, if you don't want another linear layer
    J = 10 + aaindex_pair_mat.shape[2] # change this to not be hardcoded
    print(J, "J")
    P = 6 # for now
    D = 320
    F = aaindex_res_mat.shape[1]
    print(F, "F")
    N = nheads 
    MDmis_linear_layers = LinearLayers(L, J, P, D, F, N).to(device)

    model_MDmis = MDmis(test_protein_dataset, MDmis_linear_layers, nheads, d_in,
                         d_hid, d_qkv, device = device, dropout=0.1).to(device)
    model_MDmis.load_state_dict(torch.load(f'{models_dir}_best_model.pt'))

    
    print("Check 1")

    # Then evaluate your model
    path_to_save = "/home/az2798/IDR_cons/results/"
    loss, auroc, auprc = evaluate(model_MDmis, test_loader, criterion, path_to_save,
                                  "IDRome", device=device)
    
    #test_protein_dataset()
    #test_linear()

if __name__ == "__main__":
    main()
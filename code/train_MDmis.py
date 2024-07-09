import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
#from torch.nn.utils.rnn import pad_sequence
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

###Date: 06/10/24
###Purpose: To define MDmis and train MDmis. MDmis predicts missense variant pathogenicity
###using features derived from molecular dynamics simulations and AAindex (physiochemical features).

## Defining paths
data_dir = "/share/vault/Users/az2798/"
protein_seq_mapping_dir = "/home/az2798/IDR_cons/data/"


class MDmis(nn.Module):

    def __init__(self, protein_dataset, linear_layers, nhead, d_in, d_hid, d_qkv, device, dropout=0.1):
        super(MDmis, self).__init__()
        self.protein_dataset = protein_dataset
        self.linear_layers = linear_layers
        self.sigmoid = nn.Sigmoid()
        self.nhead = nhead
        self.multi_head_attention = MultiHeadAttentionWithRoPE(nhead, d_in, d_hid, d_qkv, dropout)
        self.device = device

    def forward(self, batch):
        difference_embeddings, pair_embeddings, attention_masks, variant_matrices, md_res_tensors, md_pair_tensors = batch
        
        #padded_sequence = list(padded_sequence)
        
        #print_memory_usage("Before one forward pass")
        

        #residue_embeddings = AAindex_residue_embedding(self.protein_dataset.aa_res_matrix,
        #                                            padded_one_hot_sequences)
        #difference_embeddings = AAresidue_difference_embedding(self.protein_dataset.aa_res_matrix,
        #                                                       residue_embeddings)
        #pair_embeddings = AAindex_pair_embedding(self.protein_dataset.aa_pair_matrix,
        #                                        self.protein_dataset.aa_to_index,
        #                                        self.protein_dataset.aa_order,
        #                                        padded_sequence)
        
        #print_memory_usage("After one forward pass")

        variant_matrices = variant_matrices.to(self.device)
        difference_embeddings = difference_embeddings.to(self.device)
        pair_embeddings = pair_embeddings.to(self.device)
        md_res_tensors = md_res_tensors.to(self.device)
        md_pair_tensors = md_pair_tensors.to(self.device)
        attention_masks = attention_masks.to(self.device)

        #print(md_res_tensors.shape, "MD Res tensors")
        assert not torch.isnan(md_res_tensors).any().item(), \
            f"NA found in MD Res tensor"
        # Initialize list to hold transformed pair embeddings for each item in the batch
        #print(pair_embeddings.shape, "Pair embedding shape")
        #print(md_pair_tensors.shape, "MD pair tensors")
        # Process pair embeddings for each item in the batch
        
        concatenated_pair_embedding = torch.cat((pair_embeddings, md_pair_tensors), dim=-1)
        assert not torch.isnan(concatenated_pair_embedding).any().item(), \
            f"NA found in concatendated pair embedding"

        transformed_pair_embedding = self.linear_layers.transform_LxLxJ_to_LxLxP(concatenated_pair_embedding)
        
        assert not torch.isnan(transformed_pair_embedding).any().item(), \
            f"NA found in transformed pair embedding"

        ## Process MD residue tensors to get Attention Maps and output

        # Compute bias by summing the concatenated pair embeddings element-wise along the final axis
        attention_bias = torch.sum(transformed_pair_embedding, dim=-1, keepdim=True)
        attention_bias = attention_bias.squeeze(-1) 
        assert not torch.isnan(attention_bias).any().item(), \
            f"NA found in attention bias"
        #print(attention_masks.shape, "Attention Mask")
        #print(attention_bias.shape, "Attention Bias")
        # Apply the multi-head attention with RoPE
        attention_mask_bias =  torch.add(attention_masks, attention_bias).unsqueeze(1).repeat(1, self.nhead, 1, 1)
        attention_mask_bias = attention_mask_bias.to(self.device)
        assert not torch.isnan(attention_mask_bias).any().item(), \
            f"NA found in attention mask + bias"
        #print(attention_mask_bias)
        attention_outputs = self.multi_head_attention(md_res_tensors, attention_mask_bias)  # adding the bias into the bias
        # coerces the attention map to have additional biophysical meaning/constraint
        assert not torch.isnan(attention_outputs).any().item(), \
            f"NA found in attention output"
        
        # Process attention outputs using three linear layers
        transformed_LxD = self.linear_layers.transform_LxL_to_LxD(attention_outputs)  # LxL to LxD
        assert not torch.isnan(transformed_LxD).any().item(), \
            f"NA found in transformed LxD"
        transformed_LxDx20 = self.linear_layers.transform_LxD_to_LxDx20(transformed_LxD) # LxD to LxDx20
        assert not torch.isnan(transformed_LxDx20).any().item(), \
            f"NA found in transformed LxDx20"

        # Augment into transformed attention outputs the "concept" of missense variation
        
        transformed_difference_embedding = self.linear_layers.transform_LxFx20_to_LxDx20(difference_embeddings)
        

        # Element-wise summation of LxDx20 transformed attention outputs and LxDx20 transformed difference embeddings
        
        missense_embedding = transformed_LxDx20 + transformed_difference_embedding  # Element-wise summation
        pathogenicity_embedding = self.linear_layers.transform_LxDx20_to_Lx20(missense_embedding)
        pathogenicity_probabilities = self.sigmoid(pathogenicity_embedding)

        return pathogenicity_probabilities, variant_matrices
    
    def normalize_and_combine_embeddings(self, pair_embeddings, md_features):
        """
        Normalizes each LxL matrix independently and combines them into a single LxLx(10+H) tensor.
        
        Parameters:
        pair_embeddings (numpy.ndarray): An L x L x H matrix representing the pair embeddings.
        md_features (numpy.ndarray): An L x L x 10 matrix representing the MD simulation features.
        
        Returns:
        numpy.ndarray: An L x L x (10+H) tensor with each LxL matrix normalized independently.
        """
        # Ensure the inputs are NumPy arrays
        pair_embeddings = np.array(pair_embeddings)
        md_features = np.array(md_features)
        
        # Normalize pair_embeddings along the last dimension
        mean_pair_embeddings = np.mean(pair_embeddings, axis=(0, 1), keepdims=True)
        std_pair_embeddings = np.std(pair_embeddings, axis=(0, 1), keepdims=True)
        normalized_pair_embeddings = (pair_embeddings - mean_pair_embeddings) / std_pair_embeddings
        
        # Normalize md_features along the last dimension
        mean_md_features = np.mean(md_features, axis=(0, 1), keepdims=True)
        std_md_features = np.std(md_features, axis=(0, 1), keepdims=True)
        normalized_md_features = (md_features - mean_md_features) / std_md_features
        
        return normalized_pair_embeddings, normalized_md_features

class MisLoss(nn.Module):
    def __init__(self, weight=None):
        super(MisLoss, self).__init__()
        self.weight = weight

    def forward(self, probabilities, variant_matrices):
        """
        Compute the custom cross-entropy loss.

        Parameters:
        probabilities (List[torch.Tensor]): List of logits tensors, each of shape (20,).
        variant_matrices (List[torch.Tensor]): List of variant matrices tensors, each of shape (20,).

        Returns:
        torch.Tensor: The custom cross-entropy loss.
        """
        loss = 0.0
        # Compute the weights based on the variant matrix
        if self.weight is None:

            # Default weights: 1 for class 0, 1 for class 1
            weight = torch.ones_like(variant_matrices)
        else:

            # Custom weights provided
            weight = torch.where(variant_matrices == 0, self.weight[0], self.weight[1])

        # Compute the cross-entropy loss only for variants with labels (0 or 1)
        labeled_indices = (variant_matrices == 0) | (variant_matrices == 1)
        loss += F.binary_cross_entropy(probabilities[labeled_indices], variant_matrices[labeled_indices], weight=weight[labeled_indices])

        
        # Normalize the loss by the number of labeled variants across all variants
        #num_labeled_variants = sum(torch.sum((vm == 0) | (vm == 1)) for vm in variant_matrices)
        #loss /= num_labeled_variants
        
        return loss
    
def init_weights(m):
    """
    Initialize the weights of the given layer m.

    This function will apply Xavier uniform initialization for Linear layers,
    and set biases to 0.01. You can customize it for other types of layers as needed.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, path_to_save,
        num_epochs=10, 
        device='cuda'):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_weights = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 20 == 0:
                print(f'Training Batch index: {batch_idx}')
            # Move batch to device
            #batch = [item.to(device) for item in batch]
            optimizer.zero_grad()
            probabilities, variant_matrices = model(batch) 
            #print(probabilities, "Training Probabilities")
            #print(variant_matrices, "Variant Matrix")
            # Compute loss
            loss = criterion(probabilities, variant_matrices)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Step the learning rate scheduler
            running_train_loss += loss.item()
        # Calculate average training loss for the epoch
        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx % 20 == 0:
                print(f'Validation Batch index: {batch_idx}')
            # Move batch to device
            #batch = [item.to(device) for item in batch]
            probabilities, variant_matrices = model(batch) 
            
            # Compute loss
            loss = criterion(probabilities, variant_matrices)
            running_val_loss += loss.item()

        # Calculate average validation loss for the epoch
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f'Training Loss: {epoch_train_loss:.4f} | Validation Loss: {epoch_val_loss:.4f}')

        # Save best model weights
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = model.state_dict()
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'{path_to_save}_{epoch}.pt')
            np.save( f'{path_to_save}_{epoch}_trainingloss.npy', np.array(train_losses) )
            np.save( f'{path_to_save}_{epoch}_validationloss.npy', np.array(val_losses) )

    # Save final values and best model

    np.save( f'{path_to_save}_final_trainingloss.npy', np.array(train_losses) )
    np.save( f'{path_to_save}_final_validationloss.npy', np.array(val_losses) )
    torch.save(best_model_weights, f'{path_to_save}_best_model.pt')

    return best_model_weights, train_losses, val_losses


def main():
    #define device
    gpu_id = 1
    gpu = f'cuda:{gpu_id}'
    device = torch.device(gpu)
    torch.cuda.empty_cache()

    warmup_steps = 1000
    peak_lr = 0.001
    criterion = MisLoss([0.3, 0.7])
    L2_decay = 0.01

    #Data
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    max_seq_len = 128
    stride = 64
    batch_size = 100
    mapped_proteins = pd.read_csv("/home/az2798/IDR_cons/data/mapped_protein_seqs.csv", 
                                  index_col=0)
    mapped_proteins.rename(columns = {"sstart": "start", "send": "end", "UniProtID": "protein_id"}, inplace=True)
    mapped_proteins = mapped_proteins[mapped_proteins["source"]=="IDRome"]
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

    # Split before loading
    train_proteins, val_proteins = train_test_split(subset_mapped_proteins, test_size=0.1, random_state=42)

    print(f"Length of train_proteins: {len(train_proteins)}")
    print(f"Length of val_proteins: {len(val_proteins)}")



    train_protein_dataset = ProteinDataset(train_proteins, aaindex_res_mat, aaindex_pair_mat, clinvar_df,
                                        res_data, pair_data, max_seq_len, stride, aa_order)
    
    print_memory_usage("End of train protein dataset creation")

    val_protein_dataset = ProteinDataset(val_proteins, aaindex_res_mat, aaindex_pair_mat, clinvar_df,
                                        res_data, pair_data, max_seq_len, stride, aa_order)


    train_loader = DataLoader(train_protein_dataset, batch_size = batch_size,
                              shuffle=True) 
    
    print_memory_usage("After loading training data")

    val_loader = DataLoader(val_protein_dataset, batch_size = batch_size
                             ) 

    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")


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
    
    model_MDmis = MDmis(train_protein_dataset, MDmis_linear_layers, nheads, d_in,
                         d_hid, d_qkv, device = device, dropout=0.1).to(device)
    model_MDmis.apply(init_weights)
    #subset
    num_epochs = 60
    total_steps = len(train_loader) * num_epochs
    optimizer = AdamW(model_MDmis.parameters(), lr=peak_lr, betas=(0.9, 0.999),
                             eps=1e-8, weight_decay=L2_decay)
    scheduler = WarmupCosineDecaySchedule(optimizer, warmup_steps, peak_lr, total_steps)
    
    print("Check 1")
    #define model
    # Then train your model
    path_to_save = "/home/az2798/IDR_cons/intermediate_models/"
    best_model_weights, train_losses, val_losses = train(model_MDmis, train_loader, val_loader, criterion, optimizer, scheduler,
                                                path_to_save,
                                                num_epochs=num_epochs, device=device)
    
    #test_protein_dataset()
    #test_linear()

if __name__ == "__main__":
    main()
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from rotary_embedding_torch import RotaryEmbedding
import cProfile
import math
import psutil
import pandas as pd
import h5py
import numpy as np

###Date: 05/30/24
###Purpose: To define modules and functions to use for model training of MDmis. MDmis predicts missense variant pathogenicity
###using features derived from molecular dynamics simulations and AAindex (physiochemical features).

## Defining paths
data_dir = "/share/vault/Users/az2798/"
protein_seq_mapping_dir = "/home/az2798/IDR_cons/data/"



import numpy as np
import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    """
    Protein Dataset for handling variable-length sequences with padding and attention masking using a sliding window.

    Args:
        sequences (list of str): List of protein sequences.
        aa_res_matrix (np.ndarray): Amino acid residue matrix for embedding.
        aa_pair_matrix (np.ndarray): Amino acid pair matrix for embedding.
        max_len (int): Maximum length of the sequences. Default is 256.
        stride (int): Stride for the sliding window. Default is 128.
        variant_table (pd.DataFrame): DataFrame containing variant information.
        res_data (dict of tensors): Dictionary containing residue level MD features
        pair_data (dict of tensors): Dictionary containing pair level MD features
        max_len (int): Maximum sequence length
        stride (int): Stride for overlapping sliding window
        aa_order (str): Order of amino acids (in aaindex) as a string of length 20.

        
    Methods:
        __len__():
            Returns the number of sequences in the dataset.
        
        __getitem__(idx):
            Returns the padded sequence, attention mask, and other necessary tensors for the sequence at index idx.
    """
    def __init__(self, sequences, aa_res_matrix, aa_pair_matrix, variant_table, res_data, pair_data,
                  max_len=256, stride=128, aa_order="ARNDCQEGHILKMFPSTWYV"):
        self.sequences = sequences
        self.aa_res_matrix = aa_res_matrix
        self.aa_pair_matrix = aa_pair_matrix
        self.variant_table = variant_table
        self.max_len = max_len
        self.stride = stride
        self.aa_order = aa_order
        self.aa_to_index = {aa: idx for idx, aa in enumerate(aa_order)}
        self.precomputed_pair_embedding = self._precompute_pair_embedding()
        self.res_data = res_data
        self.pair_data = pair_data
        self.flattened_data = self._flatten_data()


    def _flatten_data(self):

        flattened_data = []
        for idx in range(len(self.sequences)):
            sequence_info = self.sequences.iloc[idx]
            sequence = sequence_info['sequence']
            start = sequence_info.get('start', 1) if 'start' in sequence_info else 0
            end = sequence_info.get('end', len(sequence)) if 'end' in sequence_info else len(sequence)
            protein_id = sequence_info.get('protein_id', '') if 'protein_id' in sequence_info else ''

            if f'{protein_id}_{start}_{end}' in self.res_data.keys():
                padded_sequence_chunks, padded_one_hot_chunks, attention_masks, md_res_tensors, md_pair_tensors = self._process_sequence(sequence, f'{protein_id}_{start}_{end}')

                variant_matrices = self._create_variant_matrices(padded_one_hot_chunks, start, end, protein_id)

                residue_embeddings = self.AAindex_residue_embedding(padded_one_hot_chunks)


                difference_embeddings = self.AAresidue_difference_embedding(residue_embeddings)

                pair_embeddings = self.AAindex_pair_embedding(padded_sequence_chunks)

                for i in range(len(padded_sequence_chunks)):
                    data_point = (difference_embeddings[i], pair_embeddings[i],
                                attention_masks[i], variant_matrices[i],
                                md_res_tensors[i], md_pair_tensors[i])
                    flattened_data.append(data_point)

        return flattened_data
    
    def __len__(self):
        return len(self.flattened_data)

    def __getitem__(self, idx):
        return self.flattened_data[idx]
    
    def _process_sequence(self, sequence, protein_start_end):
        """
        Processes a single sequence and MD data into chunks of max_len with a sliding window and creates attention masks.
        """
        chunks = []
        md_res_tensors = []
        md_pair_tensors = []
        for i in range(0, len(sequence), self.stride):
            chunk_seq = sequence[i:i+self.max_len]
            res_tensor = self.res_data[protein_start_end][i:i+self.max_len, :]
            pair_tensor = self.pair_data[protein_start_end][i:i+self.max_len, i:i+self.max_len, :]

            # Padding the res_tensor
            if res_tensor.shape[0] < self.max_len:
                padding = torch.zeros(self.max_len - res_tensor.shape[0], res_tensor.shape[1])
                res_tensor = torch.cat((res_tensor, padding), dim=0)
            md_res_tensors.append(res_tensor)

            # Padding the pair_tensor
            if pair_tensor.shape[0] < self.max_len:
                pad_size = self.max_len - pair_tensor.shape[0]
                padding_0 = torch.zeros(pad_size, pair_tensor.shape[1], pair_tensor.shape[2])
                pair_tensor = torch.cat((pair_tensor, padding_0), dim=0)
            if pair_tensor.shape[1] < self.max_len:
                pad_size = self.max_len - pair_tensor.shape[1]
                padding_1 = torch.zeros(pair_tensor.shape[0], pad_size, pair_tensor.shape[2])
                pair_tensor = torch.cat((pair_tensor, padding_1), dim=1)
            md_pair_tensors.append(pair_tensor)
            chunks.append(chunk_seq)

        padded_sequence_chunks = []
        padded_one_hot_chunks = []
        attention_masks = []

        for chunk in chunks:
            padded_sequence_chunk, padded_one_hot_chunk = self._pad_sequence(chunk)
            attention_mask = self._create_attention_mask(len(chunk))
            padded_sequence_chunks.append(padded_sequence_chunk)
            padded_one_hot_chunks.append(padded_one_hot_chunk)
            attention_masks.append(attention_mask)
   
        padded_one_hot_chunks = torch.stack(padded_one_hot_chunks)
        attention_masks = torch.stack(attention_masks)
        md_res_tensors = torch.stack(md_res_tensors)
        md_pair_tensors = torch.stack(md_pair_tensors)
        return padded_sequence_chunks, padded_one_hot_chunks, attention_masks, md_res_tensors, md_pair_tensors

    def _precompute_pair_embedding(self):
        """
        Precompute pair embeddings for all possible amino acid pairs.

        This method iterates through all possible combinations of amino acids defined
        in `self.aa_order` and computes their pair embeddings using `self.aa_pair_matrix`.
        The computed embeddings are stored in a dictionary where the keys are tuples of
        amino acid pairs and the values are the corresponding embeddings.

        Returns:
        --------
        dict
            A dictionary where keys are tuples of amino acid pairs (e.g., ('A', 'R')) 
            and values are the corresponding embeddings from `self.aa_pair_matrix`.
        """

        precomputed_pair_embeddings = {}

        for aa1 in self.aa_order:
            for aa2 in self.aa_order:
                idx1 = self.aa_to_index[aa1]
                idx2 = self.aa_to_index[aa2]
                precomputed_pair_embeddings[(aa1, aa2)] = self.aa_pair_matrix[idx1, idx2, :]
            
        return precomputed_pair_embeddings
    def _pad_sequence(self, sequence):
        """
        Pads the sequence to max_len.

        Args:
            sequence (str): The protein sequence chunk.

        Returns:
            str: Padded sequence
            torch.Tensor: Padded one hot sequence.
        """
        pad_length = self.max_len - len(sequence)
        padded_sequence = sequence + 'X' * pad_length  # Assuming 'X' is used for padding
        one_hot_sequence = self._one_hot_encode(padded_sequence)
        return padded_sequence, one_hot_sequence

    def _one_hot_encode(self, sequence):
        """
        One-hot encodes the sequence.

        Args:
            sequence (str): The protein sequence.

        Returns:
            torch.Tensor: One-hot encoded sequence of shape (max_len, 20).
        """
        one_hot = torch.zeros(self.max_len, len(self.aa_order))
        for i, aa in enumerate(sequence):
            if aa in self.aa_to_index:
                one_hot[i, self.aa_to_index[aa]] = 1.0
        return one_hot

    
    def _create_attention_mask(self, seq_len, masking_val = -1e-20):
        """
        Creates an attention mask for the sequence.

        Args:
            seq_len (int): The length of the actual sequence (before padding).

        Returns:
            torch.Tensor: Attention mask of shape (max_len, max_len).
        """
        mask = torch.full((self.max_len, self.max_len), masking_val)
        mask[:seq_len, :seq_len] = 0
        return mask

    def _create_variant_matrices(self, sequence_chunks, start, end, protein_id):
        """
        Creates variant matrices for each sequence chunk.

        Args:
            sequence_chunks (torch.Tensor): Padded sequence chunks.
            start (int): Start position of the sequence in the protein.
            end (int): End position of the sequence in the protein.
            protein_id (str): UniProt ID of the protein.

        Returns:
            list of torch.Tensor: List of variant matrices for each chunk.
        """
        num_chunks, _, _ = sequence_chunks.shape
        variant_matrices = []

        for chunk_idx in range(num_chunks):
            variant_matrix = torch.full((self.max_len, len(self.aa_order)), -1.0)  # Initialize with -1
            chunk_start = start + chunk_idx * self.stride
            chunk_end = min(chunk_start + self.max_len, end)

            relevant_variants = self.variant_table[(self.variant_table['protein_id'] == protein_id) &
                                                (self.variant_table['location'] >= chunk_start) &
                                                (self.variant_table['location'] < chunk_end)]

            for _, variant in relevant_variants.iterrows():
                loc = variant['location'] - chunk_start 
                changed_aa_idx = self.aa_to_index.get(variant['changed_residue'], -1)
                if changed_aa_idx >= 0:
                    variant_matrix[loc, changed_aa_idx] = variant['outcome']  # Use the score directly
            
            variant_matrices.append(variant_matrix)

        return variant_matrices
    
    def AAindex_residue_embedding(self, one_hot_sequence):
        """
        Computes the residue embedding for a given one-hot encoded protein sequence.
        
        Parameters:
        one_hot_sequence (torch.Tensor): A tensor of shape (batch_size, max_len, 20) representing one-hot encoded protein sequence.
        
        Returns:
        torch.Tensor: A tensor of shape (batch_size, max_len, F) representing the residue embeddings.
        """
        one_hot_sequence = one_hot_sequence.numpy()  # Convert to NumPy array
        residue_embeddings = np.dot(one_hot_sequence, self.aa_res_matrix)  # Matrix multiplication
        return torch.tensor(residue_embeddings)

    def AAresidue_difference_embedding(self, residue_embeddings):
        """
        Computes the difference in residue embeddings for each possible amino acid substitution.
        
        Parameters:
        residue_embeddings (torch.Tensor): A tensor of shape (chunks, max_len, F) representing the residue embeddings.
        
        Returns:
        torch.Tensor: A tensor of shape (max_len, F, 20) representing the difference in residue embeddings for each substitution.
        """
        residue_embeddings = residue_embeddings.numpy()  # Convert to NumPy array
        
        num_chunks, L, F = residue_embeddings.shape
        num_aa = self.aa_res_matrix.shape[0]  # Number of amino acids, should be 20
        
        difference_embeddings = np.zeros((num_chunks, L, F, num_aa))  # Initialize tensor
        for chunk_idx in range(num_chunks):
            for i in range(L):
                for j in range(num_aa):
                    new_embedding = self.aa_res_matrix[j]
                    difference_embeddings[chunk_idx, i, :, j] = new_embedding - residue_embeddings[chunk_idx, i, :]
        
        return torch.tensor(difference_embeddings) 

    def AAindex_pair_embedding(self, sequence):
        """
        Computes the pair embedding for a given protein sequence or sequence chunks.
        
        Parameters:
        sequence (list of str or torch.Tensor): A list of protein sequences as a string or one hot encoded sequence chunks as a tensor.
        
        Returns:
        torch.Tensor: A tensor of shape (batch_size, max_len, max_len, H) for sequence batches
                    or for a list of padded sequences, representing the pair embeddings.
        """
        if isinstance(sequence, list):
            # Handle single sequence input
            num_chunks, max_len = len(sequence), len(sequence[0])
            H = self.aa_pair_matrix.shape[2]  # Number of features in the pair matrix
            
            pair_embeddings = np.zeros((num_chunks, max_len, max_len, H), dtype = np.float32)
            for chunk_idx in range(num_chunks):
                for i in range(max_len):
                    aa_i = sequence[chunk_idx][i]
                    if aa_i != "X":
                        for j in range(max_len):
                            aa_j = sequence[chunk_idx][j]
                            if aa_j != "X":
                                pair_embeddings[chunk_idx, i, j, :] = self.precomputed_pair_embedding[(aa_i, aa_j)]
            
            return torch.tensor(pair_embeddings)
        elif isinstance(sequence, torch.Tensor):
            # Handle sequence chunks input
            num_chunks, max_len = sequence.shape[0], sequence.shape[1]
            H = self.aa_pair_matrix.shape[2]  # Number of features in the pair matrix

            pair_embeddings = np.zeros((num_chunks, max_len, max_len, H))
            
            for chunk_idx in range(num_chunks):
                chunk = sequence[chunk_idx]
                for i in range(max_len):
                    for j in range(max_len):
                        aa_i = self.aa_order[chunk[i].argmax().item()]
                        aa_j = self.aa_order[chunk[j].argmax().item()]
                        if (aa_i != "X" and aa_j != "X"):
                            idx_i = self.aa_to_index[aa_i]
                            idx_j = self.aa_to_index[aa_j]
                            pair_embeddings[chunk_idx, i, j, :] = self.aa_pair_matrix[idx_i, idx_j, :]
            
            return torch.tensor(pair_embeddings)
        else:
            raise ValueError("Input should be either a protein sequence string or a tensor of sequence chunks.")


def print_memory_usage(message=""):
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"{message} - Memory usage: {memory_info.rss / (1024 ** 2):.2f} MB")


def load_h5py_data(h5py_file):
    """
    Loads residue and pair data from h5py file.

    Returns:
        dict: Dictionary containing residue data.
        dict: Dictionary containing pair data.
    """
    res_data = {}
    pair_data = {}

    with h5py.File(h5py_file, 'r') as f:
        for key in f.keys():
            if key.endswith('_res'):
                protein_start_end = key.replace("_res", "")
                res_data[protein_start_end] = torch.tensor(f[key][:])
            elif key.endswith('_pair'):
                protein_start_end = key.replace("_pair", "")
                pair_data[protein_start_end] = torch.tensor(f[key][:])

    return res_data, pair_data
    
def test_protein_dataset():
    # Example data for testing
    sequences = pd.DataFrame([
        {"sequence": "ARND", "start": 1, "end": 4, "protein_id": "P1"},
        {"sequence": "ARNDARNDARND", "start": 1, "end": 12, "protein_id": "P2"},
        {"sequence": "ARN", "start": 1, "end": 3, "protein_id": "P3"}
    ])
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    F = 3  # Example feature dimension for residue matrix
    H = 4  # Example feature dimension for pair matrix
    max_len = 6  # Example maximum length for testing
    stride = 4  # Example stride for sliding window

    # Example residue and pair matrices
    aa_res_matrix = np.random.rand(20, F)
    aa_pair_matrix = np.random.rand(20, 20, H)

    # Example variant table
    variant_data = {
        'protein_id': ['P1', 'P1', 'P2', 'P2', 'P2', 'P3'],
        'location': [2, 4, 5, 9, 12, 3],
        'changed_residue': ['N', 'A', 'R', 'R', 'N', 'A'],
        'outcome': [1, 0, 1, 0, 1, 0]
    }
    variant_table = pd.DataFrame(variant_data)
    
    # Create a sample h5py file
    h5py_file = 'debug_test.h5' 
    with h5py.File(h5py_file, 'w') as f:
        for _, row in sequences.iterrows():
            protein_id = row['protein_id']
            sequence_len = len(row['sequence'])

            # Create random residue data
            residue_data = np.random.rand(sequence_len, F)
            res_key = f"{protein_id}_1_{sequence_len}_res"
            f.create_dataset(res_key, data=residue_data)

            # Create random pair data
            pair_data = np.random.rand(sequence_len, sequence_len, H)
            pair_key = f"{protein_id}_1_{sequence_len}_pair"
            f.create_dataset(pair_key, data=pair_data)
    
    # Load data from h5py file
    res_data, pair_data = load_h5py_data(h5py_file)
    
    # Create dataset
    dataset = ProteinDataset(sequences, aa_res_matrix, aa_pair_matrix, variant_table, res_data, pair_data, max_len=max_len, stride=stride, aa_order=aa_order)

    # Test __len__ method
    assert len(dataset) == sum( (len(seq['sequence']) - 1) // stride + 1 for _, seq in sequences.iterrows()), \
        f"Expected dataset length does not match, expected {sum(len(seq['sequence']) // stride + 1 for _, seq in sequences.iterrows())}, but got {len(dataset)}"

    # Test __getitem__ method
    for i in range(len(dataset)):
        (padded_chunks, attention_masks, variant_matrices, 
         difference_embeddings, pair_embeddings, 
         md_res_tensors, md_pair_tensors) = dataset[i]
        print(f"Sequence {i}:")
        print("MD res tensors:", md_res_tensors)        
        print("MD pair tensors:", md_pair_tensors)

        # Check shapes of returned tensors
        for j, chunk in enumerate(padded_chunks):
            assert chunk.shape == (max_len, len(aa_order)), \
                f"Expected padded chunk shape ({max_len}, {len(aa_order)}), but got {chunk.shape}"
        
        for j, mask in enumerate(attention_masks):
            assert mask.shape == (max_len, max_len), \
                f"Expected attention mask shape ({max_len}, {max_len}), but got {mask.shape}"

        # Check padding and attention masks
        for chunk, mask in zip(padded_chunks, attention_masks):
            original_len = int(mask[0, :].eq(0).sum().item())  # Count the number of zeros in the first row
            assert original_len <= max_len, f"Original length {original_len} exceeds max_len {max_len}"

            # Check if the sequence is correctly padded
            for j in range(original_len, max_len):
                assert chunk[j].sum().item() == 0.0, "Padding error: Non-zero values found in padded region"

            # Check if the attention mask is correct
            assert (mask[:original_len, :original_len] == 0).all(), "Attention mask error: Incorrect mask values for original sequence"
            assert (mask[original_len:, :] == -np.inf).all() and (mask[:, original_len:] == -np.inf).all(), \
                "Attention mask error: Incorrect mask values for padding"

        # Check the shapes of the variant matrices, residue embeddings, and pair embeddings
        for j, variant_matrix in enumerate(variant_matrices):
            assert variant_matrix.shape == (max_len, len(aa_order)), \
                f"Expected variant_matrix shape ({max_len}, {len(aa_order)}), but got {variant_matrix.shape}"

        assert difference_embeddings.shape[1:] == (max_len, aa_res_matrix.shape[1], len(aa_order)), \
            f"Expected difference_embeddings shape (num_chunks, {max_len}, {aa_res_matrix.shape[1]}, {len(aa_order)}), but got {difference_embeddings.shape}"

        assert pair_embeddings.shape[1:] == (max_len, max_len, aa_pair_matrix.shape[2]), \
            f"Expected pair_embeddings shape (num_chunks, {max_len}, {max_len}, {aa_pair_matrix.shape[2]}), but got {pair_embeddings.shape}"

        # Verify the shapes of md_res_tensors and md_pair_tensors
        for k, (md_res, md_pair) in enumerate(zip(md_res_tensors, md_pair_tensors)):
            assert md_res.shape == (max_len, F), \
                f"Expected md_res shape ({max_len}, {F}), but got {md_res.shape}"
            assert md_pair.shape == (max_len, max_len, H), \
                f"Expected md_pair shape ({max_len}, {max_len}, {H}), but got {md_pair.shape}"

    print("All tests passed!")

class WarmupCosineDecaySchedule(LambdaLR):
    """
    Learning Rate Scheduler with Warmup and Cosine Decay.

    This scheduler starts with a warmup phase where the learning rate increases
    linearly to a peak value, followed by a cosine decay phase where the learning 
    rate decreases. After the total number of steps, the learning rate remains constant 
    at a minimum value.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for the warmup phase.
        peak_lr (float): Peak learning rate achieved at the end of the warmup phase.
        total_steps (int): Total number of steps for the entire schedule.

    Methods:
        lr_lambda(step: int) -> float:
            Computes the learning rate multiplier for a given step based on the
            warmup and cosine decay schedule.
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, peak_lr: float, total_steps: int):
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        super(WarmupCosineDecaySchedule, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        """
        Computes the learning rate multiplier for a given step.

        Args:
            step (int): The current step in the training process.

        Returns:
            float: The learning rate multiplier for the current step.
        """
        if step < self.warmup_steps:
            decay_factor = 1 - math.cos(math.pi * float(step / self.warmup_steps) / 2)
        elif step < self.total_steps:
            decay_factor = 1 - 0.9 * math.sin(math.pi * (float(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)) / 2)  # Cosine decay
        else:
            decay_factor = 0.1
        return decay_factor * self.peak_lr



class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    This normalization method normalizes the input tensor based on its RMS norm
    along the specified dimension, then scales the normalized values using a 
    learnable parameter.

    Args:
        dim (int): Dimension of the input tensor to be normalized.
        eps (float, optional): Small epsilon value to avoid division by zero during 
            normalization. Default is 1e-8.
    """
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, dim).

        Returns:
            torch.Tensor: Normalized and scaled tensor with the same shape as input.
        """
        norm = x.norm(2, dim=-1, keepdim=True)
        return self.weight * x / (norm + self.eps)

class FeedForward(nn.Module):
    """
    Feed-Forward Neural Network with Residual Connection and RMS Normalization.

    This module implements a feed-forward network commonly used in transformer
    architectures. It consists of two linear layers with a SiLU activation in between,
    followed by a dropout layer. The output of the feed-forward network is added to
    the original input (residual connection) and then normalized using RMSNorm.

    Args:
        d_hid (int): Dimension of the input and output features.
        dropout (float): Dropout probability for regularization. Default is 0.1.

    Methods:
        forward(x):
            Applies the feed-forward network to the input tensor and returns the
            normalized output with a residual connection.
    """
    def __init__(self, d_hid, dropout=0.1):
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.SiLU(),
            nn.Linear(d_hid, d_hid),
            nn.Dropout(dropout)
        )
        self.norm = RMSNorm(d_hid)

    def forward(self, x):
        """
        Forward pass for the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_hid).

        Returns:
            torch.Tensor: Output tensor after applying the feed-forward network,
                          residual connection, and normalization.
        """
        return self.norm(x + self.ffn(x))
    
class MultiHeadAttentionWithRoPE(nn.Module):
    """
    Multi-Head Attention with Rotary Positional Embedding.

    This module implements multi-head self-attention with rotary positional
    embeddings, normalization, and a feed-forward network.

    Args:
        
        nhead (int): Number of attention heads.
        d_in (int): Dimension of the input tensor
        d_hid (int): Dimension of the model.
        d_qkv (int): Dimension of the query, key, and value vectors.
        dropout (float): Dropout probability. Default is 0.1.

    Methods:
        forward(x, mask=None):
            Applies the multi-head attention mechanism with rotary positional
            embeddings, followed by a feed-forward network.
    """
    def __init__(self, nhead, d_in, d_hid, d_qkv, dropout=0.1):
        super(MultiHeadAttentionWithRoPE, self).__init__()

        self.input_proj = nn.Linear(d_in, d_hid)

        self.model_dim = d_hid
        self.d_qkv = d_qkv
        self.head_dim = d_qkv // nhead
        self.num_heads = nhead

        self.w_qkv = nn.Linear(d_hid, d_qkv*3)

        self.rotary_emb = RotaryEmbedding(dim=self.head_dim // 2)
        self.attention_norm = RMSNorm(d_hid)
        self.feedforward = FeedForward(d_hid, dropout)
        self.output_proj = nn.Linear(d_qkv, d_qkv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for multi-head attention with RoPE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_hid).
            mask (torch.Tensor, optional): Attention mask. Default is None.

        Returns:
            torch.Tensor: Output tensor after applying multi-head attention and feed-forward network.
        """
        batch_size, seq_length, _ = x.size()
        x = self.input_proj(x)
        x = self.attention_norm(x)
        qkv = self.w_qkv(x).view(batch_size, seq_length, self.num_heads, self.head_dim*3).transpose(1, 2)
        
        assert not torch.isnan(qkv).any().item(), \
            f"NA found in qkv"
        q = self.rotary_emb.rotate_queries_or_keys(qkv[:,:,:,:self.head_dim], seq_dim=-2)
        k = self.rotary_emb.rotate_queries_or_keys(qkv[:,:,:,self.head_dim:self.head_dim*2], seq_dim=-2)
        v = qkv[:,:,:,self.head_dim*2:self.head_dim*3]
        
        

        #attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        src, tgt = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1)) 
        
        attn_bias = mask
        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        assert not torch.isnan(attn_weight).any().item(), \
            f"NA found in attention weight after adding bias"
        attn_weight = torch.log_softmax(attn_weight, dim=-1)
        assert not torch.isnan(attn_weight).any().item(), \
            f"NA found in attention weight after softmax"
        #attn_weight = torch.dropout(attn_weight, 0.1, train=True)
        attn_output = attn_weight @ v

        assert not torch.isnan(attn_output).any().item(), \
            f"NA found in scaled dot product attention output"
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_qkv)

        attn_output = self.output_proj(attn_output)

        #print(attn_output.shape, "Attention Output shape")

        x = x + self.dropout(attn_output)
        x = self.feedforward(x)

        #print(x.shape, "Final Output Shape")
        return x
class LinearLayers(nn.Module):
    def __init__(self, L, J, P, D, F, N):
        """
        Initializes the LinearLayers model with four different linear transformations.

        Parameters:
        L (int): The input dimension (length of the protein sequence).
        J (int): The dimension of the concatenated pairwise features, from AA index pairwise embeddings and MD simulations to attention maps.
        P (int): The output dimension for squeezing concatenated pairwise features to add as bias into attention map
        D (int): The output dimension of the latent embedding.
        F (int): The dimension of the residue embedding.
        N (int): The number of attention maps (should be nheads).
        """
        super(LinearLayers, self).__init__()
        self.L = L
        self.J = J
        self.P = P
        self.D = D
        self.F = F
        self.N = N
        # Linear layer to transform LxLxJ to LxLXP
        self.layer0 = nn.Linear(J, P)

        # Linear layer to transform LxLxN to LxL
        self.layer1 = nn.Linear(L * N, L)

        # Linear layer to transform LxL to LxD
        self.layer2 = nn.Linear(L, D)
        
        # Linear layer to transform LxD to LxDx20
        self.layer3 = nn.Linear(D, D * 20)
        
        # Linear layer to transform LxFx20 to LxDx20
        self.layer4 = nn.Linear(F * 20, D * 20)

        # Sigmoid layer to transform LxDx20 to Lx20
        self.layer5 = nn.Linear(D * 20, 20)

    
    
    def transform_LxLxJ_to_LxLxP(self, input_matrix):
        """
        Transforms an LxLxJ matrix to an LxLxP matrix using a linear layer.

        Parameters:
        input_matrix (torch.Tensor): An LxLxJ matrix.

        Returns:
        torch.Tensor: An LxLxP matrix.
        """
        # Ensure input is of shape LxLxJ
        #assert input_matrix.shape == (self.L, self.L, self.J)
        input_matrix = input_matrix.float()
        bsz, L, L, J = input_matrix.shape
        # Apply linear layer to each LxJ slice
        output_matrix = self.layer0(input_matrix)
        #output_matrix = output_matrix.view(self.L, self.L, self.P)

        return output_matrix
    
    def transform_LxLxN_to_LxL(self, input_matrix):
        #assert input_matrix.shape[-1] == self.N
        input_matrix = input_matrix.float()

        input_matrix = input_matrix.view(self.L, -1)

        output_matrix = self.layer1(input_matrix)

        #output_matrix = output_matrix.view(self.L, self.L)

        return output_matrix
    
    def transform_LxL_to_LxD(self, input_matrix):
        """
        Transforms an LxL matrix to an LxD matrix using a linear layer.

        Parameters:
        input_matrix (torch.Tensor): An LxL matrix.

        Returns:
        torch.Tensor: An LxD matrix.
        """
        # Ensure input is of shape LxL
        #assert input_matrix.shape == (self.L, self.L)
        input_matrix = input_matrix.float()

        # Apply linear layer to each row
        output_matrix = self.layer2(input_matrix)

        return output_matrix

    def transform_LxD_to_LxDx20(self, input_matrix):
        """
        Transforms an LxD matrix to an LxDx20 matrix using a linear layer.

        Parameters:
        input_matrix (torch.Tensor): An LxD matrix.

        Returns:
        torch.Tensor: An LxDx20 matrix.
        """
        # Ensure input is of shape LxD
        #assert input_matrix.shape == (self.L, self.D)
        input_matrix = input_matrix.float()
        bsz, L, D = input_matrix.shape

        # Apply linear layer to each row
        output_matrix = self.layer3(input_matrix)
        
        # Reshape to LxDx20

        output_matrix = output_matrix.view(bsz, self.L, self.D, 20)


        return output_matrix

    def transform_LxFx20_to_LxDx20(self, input_matrix):
        """
        Transforms an LxFx20 matrix to an LxDx20 matrix using a linear layer.

        Parameters:
        input_matrix (torch.Tensor): An LxFx20 matrix.

        Returns:
        torch.Tensor: An LxDx20 matrix.
        """
        # Ensure input is of shape LxFx20
        #assert input_matrix.shape == (self.L, self.F, 20)
        input_matrix = input_matrix.float()
        bsz, L, F, _ = input_matrix.shape
        
        # Flatten the last two dimensions
        input_matrix = input_matrix.view(bsz, self.L, -1)
        
        # Apply linear layer to each row
        output_matrix = self.layer4(input_matrix)
        
        #print(output_matrix.shape, "Fx20 to Dx20")

        output_matrix = output_matrix.view(bsz, self.L, self.D, 20)
        
        return output_matrix
    
    def transform_LxDx20_to_Lx20(self, input_matrix):
        """
        Transforms an LxDx20 matrix to an Lx20 matrix using a linear layer.

        Parameters:
        input_matrix (torch.Tensor): An LxDx20 matrix.

        Returns:
        torch.Tensor: An Lx20 matrix.
        """
        # Ensure input is of shape LxDx20
        #assert input_matrix.shape == (self.L, self.D, 20)
        input_matrix = input_matrix.float()        

        bsz, L, D, _ = input_matrix.shape
        # Flatten the last two dimensions
        input_matrix = input_matrix.view(bsz, self.L, -1)
        
        # Apply linear layer to each row
        output_matrix = self.layer5(input_matrix)

        output_matrix = output_matrix.view(bsz, self.L, 20)

        return output_matrix

def test_linear():
    L = 4   # Example input dimension
    D = 8   # Desired output dimension
    P = 4
    J = 6
    F = 563
    # Initialize model
    model = LinearLayers(L, P, J, D, F)

    # Example input matrices
    input_matrix_LxL = torch.randn(L, L)
    input_matrix_LxD = torch.randn(L, D)
    input_matrix_LxFx20 = torch.randn(L, F, 20)
    input_matrix_LxLxJ = torch.randn(L, L, J)


    # Transform the input matrices
    output_matrix_LxD = model.transform_LxL_to_LxD(input_matrix_LxL)
    output_matrix_LxDx20_from_LxD = model.transform_LxD_to_LxDx20(input_matrix_LxD)
    output_matrix_LxDx20_from_LxFx20 = model.transform_LxFx20_to_LxDx20(input_matrix_LxFx20)
    output_matrix_LxLxP = model.transform_LxLxJ_to_LxLxP(input_matrix_LxLxJ)

    # Print output shapes to verify
    print("Output shape (LxL to LxD):", output_matrix_LxD.shape)                # Should be (L, D), e.g., (4, 8)
    print("Output shape (LxD to LxDx20):", output_matrix_LxDx20_from_LxD.shape)  # Should be (L, D, 20), e.g., (4, 8, 20)
    print("Output shape (LxFx20 to LxDx20):", output_matrix_LxDx20_from_LxFx20.shape)  # Should be (L, D, 20), e.g., (4, 8, 20)
    print("Output shape (LxLxJ to LxLxP):", output_matrix_LxLxP.shape)  # Should be (L, L,P)


def main():
    test_protein_dataset()
    #test_linear()

if __name__ == "__main__":
    main()
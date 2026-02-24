import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.metrics.pairwise import cosine_similarity
import glob
from collections import Counter
from Bio import SeqIO
import math
import tqdm
import sys
import pathlib
ROOT = pathlib.Path(__file__).parents[1]
sys.path.append(str(ROOT))
from utils import *
from config import config
def cluster_AAs(AAIndex_features, n_PCs, num_clusters, clustering_method,
                aa_order, results_dir):
    pca = PCA(n_components=n_PCs)
    pca_result = pca.fit_transform(AAIndex_features)
    explained_variance = pca.explained_variance_ratio_ * 100  

    if clustering_method == 'DBSCAN':
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(pca_result)
        labels = clustering.labels_

    elif clustering_method == 'KMeans':
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(pca_result)
        labels = kmeans.labels_
    else:
        raise ValueError("Unsupported clustering method. Use 'DBSCAN' or 'KMeans'.")
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
    
    for i, aa in enumerate(aa_order):
        plt.text(pca_result[i, 0], pca_result[i, 1], aa, fontsize=12, ha='center')

    plt.title(f"Amino Acid Clustering using Top {n_PCs} Principal Components")
    plt.xlabel(f"PC 1: {explained_variance[0]:.2f}%")
    plt.ylabel(f"PC 2: {explained_variance[1]:.2f}%")
    plt.colorbar(label='Cluster Label')
    plt.savefig(
        os.path.join(results_dir, "AA_Clusters.png"), dpi = 300, bbox_inches = "tight"
    )

    plt.figure(figsize=(8, 6))
    top_n = min(5, len(explained_variance))
    plt.bar(range(1, top_n + 1), explained_variance[:top_n], color='skyblue')
    plt.xticks(range(1, top_n + 1))
    plt.title("Explained Variance of Top Principal Components")
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance (%)")
    plt.ylim(0, max(explained_variance[:top_n]) * 1.1) 
    plt.savefig(
        os.path.join(results_dir, "Explained_Variance.png"), dpi=300, bbox_inches="tight"
    )
    plt.clf()

    aa_cluster_map = {aa: label for aa, label in zip(aa_order, labels)}
    return aa_cluster_map

def compute_sigma(sequence, aa_charge_map):
    #Count the number of positives and negatives
    n_pos, n_neg = 0, 0
    for i, aa in enumerate(sequence):
        if aa in aa_charge_map:
            if aa_charge_map[aa] == "Pos":
                n_pos += 1
            else:
                n_neg += 1

    charge_diff = n_pos - n_neg
    charge_sum = n_pos + n_neg
    sigma = math.pow(charge_diff, 2) / charge_sum if (charge_diff != 0) and (charge_sum !=0) else 0

    return sigma

def compute_gamma(sequence_region, num_segments, segment_size,
                  aa_charge_map):
    overall_sigma = compute_sigma(sequence_region, aa_charge_map)
    segment_sigmas = []

    total_length = len(sequence_region)
    overlap = (segment_size * num_segments - total_length) // (num_segments - 1)

    for i in range(num_segments): #i denotes index of segment
        start = i * (segment_size - overlap)
        end = start + segment_size
        
        segment = sequence_region[start:end]
        sigma = compute_sigma(segment, aa_charge_map)
        segment_sigmas.append(sigma)
    segment_sigmas = np.array(segment_sigmas)

    gamma = np.sum(np.power(segment_sigmas - overall_sigma, 2)) / num_segments
    return gamma



def extract_LCR_polyX(s): # for each sub-sequence, check if it is an
                            # LCRs of polyX pattern 
    if s == len(s) * s[0]:
        return 1
    else:
        return 0


def extract_LCR_polyXY(s):
    counter = Counter(s)
    polyXY= True
    if len(counter.keys()) == 2:
        # only two unique AAs
        for aa in counter.keys():
            num_appearances = counter[aa]
            if num_appearances<2:
                polyXY = False
    else:
        polyXY = False
    if polyXY:
        return 1
    else:
        return 0

def find_suitable_window(sequence, index, window_size):
    if (index >= window_size//2) and (index < len(sequence) - window_size//2):
                        sequence_region = sequence[index - window_size // 2 : index] + sequence[index + 1 : index + window_size // 2 + 1]
    elif index < window_size//2:
        #Just take first window_size available
        sequence_region = sequence[0: window_size // 2] + sequence[window_size // 2 + 1 : window_size]
    else:
        #Just take final window_size
        sequence_region = sequence[len(sequence) - window_size : len(sequence) - window_size // 2 - 1] + sequence[len(sequence) - window_size // 2:]
    return sequence_region

def compute_MSA_features(variants_table, uniprot_column_name,
                             location_column_name, MSA_directory,
                             window_size, polyX_LCR_size,polyXY_LCR_size,
                             num_segments, segment_size, 
                             aa_charge_map, aa_cluster_map,
                             aa_order):
    '''
    Function that takes in variants and parses Zoonomia MSAs to compute features such as
    charge pattern, entropy (site and region), composition of amino-acids, and low-complexity regions
    using repeat recognition.
    Parameters:
    ----------
    variants_table: pandas.DataFrame
    A DataFrame containing information about protein variants, including location, sequence, 
        and metadata like "outcome", "location", etc.
    
    uniprot_column_name: str
    The column name for the UniProt ID of the protein

    location_column_name: str
    The column name for the location of the variant. Should be the exact location in the protein, rather than a 0-based index

    MSA_directory : str, Path
    The path where Zoonomia MSAs (cleaned) are stored. The top record should be the human UniProt sequence. Each file should be stored using 
    the pattern {uniprot_id}_*.fasta

    window_size: int
    An odd number that is provided to indicate how large the window should be for computing charge pattern, compositional conservation, and entropy
    across the homologous sequences in the MSA.

    polyX_LCR_size: int
    An integer for the number of consecutive residues that constitute a "repeat" -- for determining low-complexity regions of polyX (single amino-acid) 

    polyXY_LCR_size: int
    An integer for the number of consecutive residues that constitute a "repeat" -- for determining low-complexity regions of polyXY (two amino-acids repeating) 

    num_segments: int
    Works with window_size and is specifically for computing charge pattern (gamma). Done by taking sub-segments of a region and computing their charge
    in order to compute the "segregation" of charge.

    segment_size: int
    Works with window_size and num_segments for computing charge pattern. The size of each sub-segment.

    aa_charge_map: dict
    A dictionary that maps amino-acids (single letter) to their charge

    aa_cluster_map: dict
    A dictionary that maps amino-acids (single letter) to a cluster based on their properties

    aa_order: str
    Order of amino-acids used in our analysis and in the AA-index database.
    '''
    if window_size//2 ==0:
        raise ValueError("Please select an odd number for window size for symmetry.")
    #unique_transcripts = set(variants_table[ENST_column_name])
    unique_proteins = set(variants_table[uniprot_column_name])
    results = []
    for protein in tqdm.tqdm(unique_proteins):
        print(f"Processing {protein}")
        variants_for_transcript = variants_table[variants_table[uniprot_column_name] == protein]
        file_list = glob.glob(os.path.join(MSA_directory, protein + "*"))
        if len(file_list) == 0:
            #print(f"{protein} MSA is not available") 
            continue
        with open(file_list[0], "r") as input_MSA:
            #Open that MSA file
            seq_records = list(SeqIO.parse(input_MSA, "fasta"))
            
            ref_seq = seq_records[0].seq #use the reference sequence to compute the global amino-acid composition
            global_composition = np.zeros(20)
            for amino_acid in ref_seq:
                idx_ref = aa_order.find(amino_acid) #make sure it is a standard amino acid
                if idx_ref != -1:
                    global_composition[idx_ref] +=1 
            global_composition = global_composition / np.sum(global_composition)
            for i, row in variants_for_transcript.iterrows():
                index = row[location_column_name] - 1 #to account for zero indexing
                #print(index)
                if index >= len(seq_records[0]): #perhaps wrong sequence
                    print(f"{protein} MSA is not matching variant information")
                    continue 
                #now, navigate within the window_size of this index
                cluster_counts_list = []  
                residue_counts_region = np.zeros((window_size - 1, 20))  #To get the actual entropy of residues
                residue_counts_site = np.zeros(20) 
                num_STYs = 0
                gamma_list = [] #To get the charge pattern


                #use ref-seq for polyX and polyY
                polyX = extract_LCR_polyX(find_suitable_window(ref_seq, index, polyX_LCR_size))
                polyXY = extract_LCR_polyXY(find_suitable_window(ref_seq, index, polyXY_LCR_size))

                for record in seq_records:
                    sequence = record.seq
                    sequence_region = find_suitable_window(sequence, index, window_size)
                    cluster_counts = np.zeros(max(aa_cluster_map.values()) + 1)
                    for j, aa in enumerate(sequence_region):
                        if aa in aa_order:
                            cluster_counts[aa_cluster_map[aa]] += 1
                            residue_counts_region[j, aa_order.find(aa)] += 1
                    aa_msa = sequence[index]
                    if aa_msa in aa_order:
                        residue_counts_site[aa_order.find(aa_msa)] +=1 
                    
                    if aa_msa in ["S", "T", "Y"]:
                        num_STYs+=1
                    gamma = compute_gamma(sequence_region, num_segments, segment_size,
                                  aa_charge_map)

                    cluster_counts_list.append(cluster_counts)
                    gamma_list.append(gamma)
                similarity_matrix = cosine_similarity(np.array(cluster_counts_list))
                average_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                
                position_entropies = ss.entropy(residue_counts_region, base=2, axis=1)
                average_entropy_region = np.mean(position_entropies)  
                
                average_entropy_site = ss.entropy(residue_counts_site, base = 2)


                average_gamma = np.mean(np.array(gamma_list))
                std_gamma = np.std(np.array(gamma_list))
                results.append({
                        "UniProtID": protein,
                        "Location": row[location_column_name],
                        "Average Cosine Similarity": average_similarity,
                        "Average Entropy Site": average_entropy_site,
                        "Average Entropy Region": average_entropy_region,
                        "Average Gamma": average_gamma,
                        "StDev Gamma": std_gamma,
                        "Probability of STY": num_STYs/len(seq_records),
                        "Poly_X": polyX,
                        "Poly_XY": polyXY,
                        **{f"Global Composition_{aa_order[j]}": val for j, val in enumerate(global_composition)},
                    })
    return pd.DataFrame(results)
def main():
    data_dir = os.path.abspath(config["data_dir"])
    results_dir = os.path.abspath(config["results_dir"])
    MSA_dir = os.path.abspath(config["MSA_dir"])
    aa_order = "ARNDCQEGHILKMFPSTWYV"

    aa_charge_map = {"R": "Pos", "H": "Pos", "K": "Pos", "D": "Neg",
                     "E": "Neg"}

    aaindex_res = np.load(os.path.join(data_dir, "aa_index1.npy"))
    print(aaindex_res.shape)

    aa_cluster_map = cluster_AAs(aaindex_res, n_PCs=5, num_clusters=4, clustering_method='KMeans',
                aa_order=aa_order, results_dir=results_dir)
    variants_table = pd.read_csv(
        os.path.join(data_dir, "merged_proteome_information_clinical.csv"), index_col = 0
        )
    #variants_table.dropna(subset= "ENST_id", inplace=True)
    results_df = compute_MSA_features(variants_table, "UniProtID",
                             "location", MSA_dir,
                             window_size=19, polyX_LCR_size=6,polyXY_LCR_size= 10,
                             num_segments=10,segment_size= 5, aa_charge_map=aa_charge_map,
                             aa_cluster_map=aa_cluster_map, aa_order=aa_order)
    results_df.to_csv(
        os.path.join(data_dir, "sequence_composition_MSA.csv")
    )    
if __name__ == "__main__":
    main()
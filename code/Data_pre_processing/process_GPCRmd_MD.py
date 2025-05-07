import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import os
import numpy as np
from Bio import SeqIO
from Bio import pairwise2
from Bio.Seq import Seq
from utils import *
import h5py

import sys
import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.append(ROOT)
from config import config

pd.set_option('display.max_columns', 500)

def extract_uniprot_id(record_id):
    """
    Extracts the UniProt ID (e.g., 'Q9UNL2') from record IDs like 'sp|Q9UNL2|SSRG_HUMAN'.
    """
    if "|" in record_id:
        return record_id.split("|")[1] 
    return record_id


def calculate_identity(seq1, seq2):
    """Calculate percentage identity between two aligned sequences."""
    matches = sum(res1 == res2 for res1, res2 in zip(seq1, seq2) if res1 != '-' and res2 != '-')
    total = sum(1 for res1, res2 in zip(seq1, seq2) if res1 != '-' and res2 != '-')
    return (matches / total) * 100 if total > 0 else 0

def split_alignment(aligned_query, aligned_target, min_match_length=5):
    """
    Splits the alignment into segments where the query and target have continuous matches longer than min_match_length.
    """
    matches = []
    
    in_match = False
    query_pos = 0
    target_pos = 0
    for (q_res, t_res) in zip(aligned_query, aligned_target):
        aligned_query_true_length = len(aligned_query.replace("-", ""))
        if (q_res != "-") and (t_res !="-"):
            if not in_match:
                #Start of a new match
                target_start = target_pos
                query_start = query_pos
                in_match = True
        
        
        elif (q_res == "-") or (t_res == "-"):
            if in_match:
                #End of the current match
                target_end = target_pos
                query_end = query_pos

                matches.append({"query_start": query_start,
                                "query_end": query_end,
                                "target_start": target_start,
                                "target_end": target_end})
                in_match = False
            
        else:
            raise ValueError("Non standard character in alignment")
        if q_res != "-":
            query_pos+=1 #query pos only moves if we encounter a character
        target_pos +=1 #target pos always moves forward
    if in_match: #capturing a match if it was at the end
        matches.append({        "query_start": query_start,
                                "query_end": aligned_query_true_length,
                                "target_start": target_start,
                                "target_end": len(aligned_target)})
    
    filtered_matches = [
        match for match in matches if (match["target_end"] - match["target_start"] + 1) >= min_match_length
    ]
    return filtered_matches

def align_and_update_metadata(metadata_row, uniprot_sequences, min_match_length=5):
    uniprot_id = metadata_row["UniProtID"]
    sequence = metadata_row["sequence"]

    if uniprot_id not in uniprot_sequences:
        return []  

    uniprot_seq = uniprot_sequences[uniprot_id]

    #Perform pairwise alignment
    alignments = pairwise2.align.localxx(Seq(sequence), Seq(uniprot_seq))
    best_alignment = alignments[0]  
    aligned_query, aligned_target, score, start, end = best_alignment

    #Calculate identity
    identity = calculate_identity(aligned_query, aligned_target)
    if identity < 98:
        return []  

    matches = split_alignment(aligned_query, aligned_target, min_match_length)
    updated_rows = []
    for match in matches:
        updated_row = metadata_row.copy()
        updated_row["MD_start"] = match["query_start"] + 1
        updated_row["MD_end"] = match["query_end"] 
        updated_row["sstart"] = match["target_start"] + 1  #Convert to 1-based index
        updated_row["send"] = match["target_end"]  #Convert to 1-based index
        updated_row["sequence"] = metadata_row["sequence"][match["query_start"]:match["query_end"]]
        updated_row["MD_protein_start_end"] = metadata_row["protein_start_end"]
        updated_row["protein_start_end"] = f"{uniprot_id}_{match['target_start'] + 1}_{match['target_end']}"
        updated_rows.append(updated_row)

    return updated_rows


def main():
    vault_dir = os.path.abspath(config["vault_dir"])
    data_dir = os.path.abspath(config["data_dir"])

    clinvar_labels_df = pd.read_csv(os.path.join(vault_dir, "ClinVar_Data", "training.csv")) 
    MD_metadata = pd.read_csv(os.path.join(data_dir, "MD_metadata.csv"),
                                        index_col=0)

    MD_metadata.rename(columns={"source": "MD Data Source"}, inplace=True)

    GPCRmd_metadata = MD_metadata[MD_metadata["MD Data Source"] == "GPCRmd"]

    print(GPCRmd_metadata.head())
    clinvar_labels_df.rename(columns = {"score":"outcome", "uniprotID":"UniProtID", "pos.orig": "location", 
                        "alt": "changed_residue", 'data_source': 'Label Source',
                        "Ensembl_transcriptid": "ENST_id"}, inplace=True)
    clinvar_labels_df = clinvar_labels_df[["outcome", "UniProtID", "location", "changed_residue", "VarID", 
                            "Label Source", "ENST_id"]]


    uniprot_db = os.path.join(data_dir, "uniprot_db.fasta")
    uniprot_sequences = {
        extract_uniprot_id(record.id): str(record.seq) for record in SeqIO.parse(uniprot_db, "fasta")
    }
    updated_rows = []
    for _, row in GPCRmd_metadata.iterrows():
        new_rows = align_and_update_metadata(row, uniprot_sequences)
        updated_rows.extend(new_rows)

    GPCRmd_metadata_corrected = pd.DataFrame(updated_rows)
    GPCRmd_metadata_corrected.drop_duplicates(subset="protein_start_end", inplace=True)
    print(GPCRmd_metadata_corrected.tail(n=20))
    GPCRmd_metadata_corrected.to_csv(
        os.path.join(data_dir, "GPCRmd_metadata_processed.csv")
    )
if __name__ == "__main__":
    main()
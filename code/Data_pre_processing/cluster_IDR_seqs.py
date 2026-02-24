import pandas as pd
import numpy as np
import glob
from Bio import Seq, SeqRecord, SeqIO
import subprocess
import os
import sys
import pathlib
ROOT = pathlib.Path(__file__).parents[1]
sys.path.append(str(ROOT))
from config import config

data_dir = config["data_dir"]

# take all the IDRs from the MD metadata and store them as a fasta
MD_metadata = pd.read_csv(os.path.join(data_dir, "MD_metadata.csv"),
                                    index_col=0)

IDRs_only = MD_metadata[MD_metadata["source"] == "IDRome"]
#print(IDRs_only.shape)
seqio_records = []

for _,row in IDRs_only.iterrows():
    seqio_records.append(SeqIO.SeqRecord(row["sequence"], id = row["name"]))

SeqIO.write(seqio_records, os.path.join(data_dir, "IDR_clustering", "IDR_seqs.fa"), "fasta")

# now, perform mmseqs clustering on these sequences
# subprocess.run(["mmseqs", "createdb", os.path.join(data_dir, "IDR_clustering", "IDR_seqs.fa"), 
#                 os.path.join(data_dir, "IDR_clustering", "IDR_DB")])

if not os.path.exists(os.path.join(data_dir, "IDR_clustering", "IDR_DB_clu_all_seqs.fasta")):
    subprocess.run(["mmseqs", "easy-cluster", os.path.join(data_dir, "IDR_clustering", "IDR_seqs.fa"), 
                os.path.join(data_dir, "IDR_clustering", "IDR_DB_clu"),
                "/home/az2798/MDmis/tmp/", "--min-seq-id", '0.3'])


cluster_id = 0
cluster_dict = {}
with open(os.path.join(data_dir, "IDR_clustering", "IDR_DB_clu_all_seqs.fasta"), "r") as file:
    lines = file.readlines()
    for line in lines:
        if "<unknown description>" in line:
            protein_id = line.replace("<unknown description>", "").replace(">","").strip()
            cluster_dict[protein_id] = cluster_id
        elif line.startswith(">"):
            #start of a cluster
            cluster_id+=1
        else:
            #sequence
            continue
            

print(cluster_dict)
MD_metadata["Cluster"] = MD_metadata["name"].map(cluster_dict)
MD_metadata.to_csv(os.path.join(data_dir, "MD_metadata_clustered.csv"))
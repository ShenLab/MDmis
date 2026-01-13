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


data_dir = os.path.abspath(config["data_dir"])
vault_dir = os.path.abspath(config["vault_dir"])

IDRs_table = pd.read_csv(
    os.path.join(
        data_dir, "clinical_train_val", "feature_table.csv"
    ) 
    , index_col= 0, low_memory= False
)


IDRs_table["start"] = IDRs_table["protein_start_end"].str.split("_").str[1].astype(int)
IDRs_table["end"] = IDRs_table["protein_start_end"].str.split("_").str[2].astype(int)

IDRs_table["Region Length"] = IDRs_table["end"] - IDRs_table["start"] + 1

IDRs_table["Variant Effect"] = np.where(IDRs_table["outcome"] == 1, "Pathogenic", "Benign")

IDRs_table['Length Category'] = np.select(
    [
        (IDRs_table["Variant Effect"] == "Pathogenic") & 
        (IDRs_table["Region Length"] > 800),
        (IDRs_table["Variant Effect"] == "Pathogenic") &
        (IDRs_table["Region Length"] <= 800),
        IDRs_table["Variant Effect"] == "Benign"
    ],
    ['Pathogenic >800aa IDRs', 'Pathogenic <800aa IDRs', 'Benign']
)


### Subset UniProt sequences to generate a fasta for MSA generation
pathogenic_vlong = IDRs_table[IDRs_table["Length Category"]=="Pathogenic >800aa IDRs"]["UniProtID"].unique()

filtered_records = []

for record in SeqIO.parse(os.path.join(vault_dir, "IDR_MSA", "uniprot_db.fasta"), "fasta"):
    uniprot_id = record.id.split("|")[1] if "|" in record.id else record.id
    if uniprot_id in pathogenic_vlong:
        filtered_records.append(record)

SeqIO.write(filtered_records, os.path.join(vault_dir, "IDR_MSA", "vlong_IDRs.fasta"), "fasta")


subprocess.run(["python", os.path.join(config["colabfold_dir"], "MSA", "0_protocol", "colab", "search.py"), #python file
                os.path.join(vault_dir, "IDR_MSA", "vlong_IDRs.fasta"), #query fasta
                os.path.join(config["colabfold_dir"], "seqdb"), #dbbbase
                os.path.join(vault_dir, "IDR_MSA"), #output directory, aka base
                "--db1", "uniref30_mmseqs/uniref30_2302_db",
                "--db3", "cfdb_mmseqs/colabfold_envdb_202108_db",  #databases to search
                "--use-env", "1", "--unpack-targets", "uniref", "merge" #the targets to unpack
                ])
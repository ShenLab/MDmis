import pronto
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from utils import *

def retrieve_third_parent(ontology, term):

    if term in ontology.terms():
        all_ancestors = list(ontology[term].superclasses())
        if len(all_ancestors) > 3:
            return all_ancestors[-3].name
    return None
    

def load_GO_annotations(filepath):

    column_names = ["Source", "UniProtID", "Gene_Name", "Role", "GO_Term"]

    with open(filepath, 'r') as file:
        skiprows = sum(1 for line in file if line.startswith('!'))

    df = pd.read_csv(filepath, sep="\s+",
                     skiprows=skiprows, 
                     usecols=range(5), header=None)
    df.columns = column_names
    return df

def main():
    vault_dir = "/share/vault/Users/az2798/GO_annotation/"
    go_annotations = load_GO_annotations(
        os.path.join(vault_dir, "goa_human.gaf")
    )

    go_annotations = go_annotations[go_annotations["Role"].isin(
        ["enables", "involved_in"]
    )]

    print(go_annotations.head())

    GO_ontology = pronto.Ontology(
        os.path.join(vault_dir, "go-basic.obo")
    )

    go_annotations["Third_Parent"] = go_annotations["GO_Term"].apply(
        lambda x: retrieve_third_parent(GO_ontology, x))

    print(go_annotations.head())

    go_annotations.to_csv(
        os.path.join(vault_dir, "processed_go_annotations.csv")
    )
if __name__ == "__main__":
    main()
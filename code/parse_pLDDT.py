import pandas as pd
import numpy as np
import glob
import re
pd.set_option('display.max_columns', 500)


def process_scores(pLDDT_file):
    data = []
    with open(pLDDT_file, 'r') as file:
        for line in file:
            uniprot_id, scores_str = line.split()
            scores = scores_str.split(',')    
            
            for location, score in enumerate(scores, start=1):
                data.append({
                    'UniProtID': uniprot_id,
                    'location': location,
                    'pLDDT': float(score)
                })

    df = pd.DataFrame(data)
    
    return df

def main():
    
    pLDDT_file_path = "/home/az2798/IDR_cons/data/pLDDT_scores/9606.pLDDT.tdt" 


    pLDDT_scores = process_scores(pLDDT_file_path)
    pLDDT_scores.to_csv("/home/az2798/IDR_cons/data/pLDDT_scores/processed_pLDDT_scores.csv")

if __name__ == "__main__":
    main()

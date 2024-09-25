import numpy as np
import pandas as pd
import os
from Bio import SeqIO
import glob
codon_to_aa = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    'NNN': '*'
}

# A comprehensive map of codons to Amino acids -- NNN typically marks the stop


def translate_codon_alignment(seq_record):
    """
    Translates a nucleotide sequence to an amino acid sequence.
    
    Parameters:
    seq_record (SeqRecord): A SeqRecord object containing the codon sequence.
    
    Returns:
    SeqRecord: A SeqRecord object with the translated amino acid sequence.
    """
    codon_seq = seq_record.seq  #Get the codon sequence from the SeqRecord
    protein_seq = codon_seq.translate(to_stop=False)  #Translate codon to amino acid sequence - NNN will likely be marked as X (stop)
    return SeqIO.SeqRecord(protein_seq, id=seq_record.id, description="translated")

def convert_to_aa_msa(input_file, output_file):
    """
    Converts a codon alignment in .fa format to an amino acid MSA using SeqIO.
    
    Parameters:
    input_file (str): Path to the input .fa codon alignment file.
    output_file (str): Path to the output file where the amino acid MSA will be written.
    """
    #Read the input file using SeqIO
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        seq_records = SeqIO.parse(infile, "fasta")  
        translated_records = (translate_codon_alignment(record) for record in seq_records)  
        #Translate each record and then write them to the output file
        SeqIO.write(translated_records, outfile, "fasta")


def main():
    codon_dir = "/share/vault/Users/az2798/Zoonomia/codon_alignments/"
    protein_msa_dir = "/share/vault/Users/az2798/Zoonomia/protein_alignments/"
    list_of_fastas = glob.glob(os.path.join(codon_dir, "*.fasta"))

    #Perform this for each file
    for fasta in list_of_fastas:
        #get basename and use that as outfile
        base_name = os.path.basename(fasta)
        output_file = os.path.join(protein_msa_dir, base_name)
        print(f"Processing {fasta} -> {output_file}")
        convert_to_aa_msa(fasta, output_file)

if __name__ == "__main__":
    main()
import csv

data_dir = "/home/az2798/IDR_cons/data/"
input_csv = 'protein_sequences.csv'
output_fasta = 'protein_sequences.fasta'
linker = "<linker>"
with open(f'{data_dir}{input_csv}', 'r') as csvfile, open(f'{data_dir}{output_fasta}', 'w') as fastafile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        name = row['name']
        sequence = row['sequence']
        peptides = sequence.split(linker)
        
        #Write each peptide as a separate FASTA entry
        for i, peptide in enumerate(peptides):
            entry_name = f"{name}_part{i+1}" if len(peptides) > 1 else name
            fastafile.write(f'>{entry_name}\n')
            fastafile.write(f'{peptide}\n')

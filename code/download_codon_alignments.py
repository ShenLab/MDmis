import os
base_url = "https://genome.senckenberg.de/download/TOGA/human_hg38_reference/MultipleCodonAlignments/"
with open('/share/vault/Users/az2798/Zoonomia/list_of_transcripts', 'r') as file:
    lines = file.readlines()
lines = lines[:-1]

for line in lines:
    filename = line.strip()
    url = base_url + filename  
    wget_command = f"wget --no-check-certificate {url} -P /share/vault/Users/az2798/Zoonomia/codon_alignments/"
    os.system(wget_command)

print("Download complete.")

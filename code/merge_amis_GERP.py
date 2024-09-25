import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

pd.set_option("display.max_columns", 100)

###Define paths
results_dir = "/home/az2798/IDR_cons/results/"
data_dir = "/home/az2798/IDR_cons/data/"
GERP_path = "/share/vault/Users/az2798/GERP_data/dbNSFP4_merged.csv"
proteins_clinvar_mapped  = f'{data_dir}proteins_clinvar_mapped.csv'

#Process proteins_clinvar_mapped

proteins_clinvar_mapped = pd.read_csv(proteins_clinvar_mapped, index_col=0)

proteins_clinvar_mapped["location"] = proteins_clinvar_mapped["location"].astype(str)
unique_proteins_in_clinvar = proteins_clinvar_mapped["protein_id"].unique()

print(len(unique_proteins_in_clinvar), "Unique Proteins in Clinvar")
####
#Process Amis data
alpha_missense_table = pd.read_csv(f'{data_dir}AlphaMissense_data/AlphaMissense_aa_substitutions.tsv', sep='\t',
                                   low_memory=False, skiprows=[0,1,2])

print("Loaded Amis data",alpha_missense_table.head())


alpha_missense_table = alpha_missense_table[alpha_missense_table["uniprot_id"].isin(unique_proteins_in_clinvar)]

alpha_missense_table['location_amis'] = alpha_missense_table['protein_variant'].str.extract(r'(\d+)')

alpha_missense_table['location_amis'] = alpha_missense_table['location_amis'].astype(str)

alpha_missense_table['changed_aa_amis'] = alpha_missense_table['protein_variant'].str.extract(r'([A-Z])$', expand=False)

print("Parsed data", alpha_missense_table.head())

alpha_missense_merged_clinvar = pd.merge(left=proteins_clinvar_mapped, right=alpha_missense_table,
                                        left_on= ["protein_id", "location", "changed_residue"],
                                        right_on= ["uniprot_id", "location_amis", "changed_aa_amis"],
                                        how="left")

print(alpha_missense_merged_clinvar.shape, "Amis Merging Shape")
print(alpha_missense_merged_clinvar["am_pathogenicity"].isnull().sum())
print(alpha_missense_merged_clinvar[alpha_missense_merged_clinvar["am_pathogenicity"].isna()])


####
def split_rows(df, columns_to_split):
    #Split the specified columns by ';' and create a new dataframe by exploding the lists
    split_df = df.assign(**{col: df[col].str.split(';') for col in columns_to_split})
    exploded_df = split_df.explode(columns_to_split, ignore_index=True)
    return exploded_df

#Process dbNSFP data

GERP_df = pd.read_csv(GERP_path, index_col=0, header=0)
print(len(GERP_df["Uniprot_acc"].unique()), "Unique Proteins in GERP")
columns_to_split = ['aapos', 'Ensembl_transcriptid', 'Uniprot_acc']

GERP_df_split = split_rows(GERP_df, columns_to_split)
print(GERP_df_split.head(20))

amis_GERP_df = pd.merge(left=alpha_missense_merged_clinvar,
                        right=GERP_df_split, left_on=["protein_id", "location"],
                        right_on=["Uniprot_acc", "aapos"],
                        how="left")

amis_GERP_df.drop_duplicates(subset= ["VarID"], inplace=True)
print(amis_GERP_df.shape, "Amis GERP Shape")
print(amis_GERP_df[amis_GERP_df["GERP++_RS"].isna()])

amis_GERP_df.to_csv(f"{data_dir}amis_GERP_clinvar.csv")


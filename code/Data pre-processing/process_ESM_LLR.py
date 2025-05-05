import pandas as pd
import glob
import os


vault_dir = "/nfs/user/Users/az2798/ESM1b_data/content/ALL_hum_isoforms_ESM1b_LLR/"
output_dir = "/nfs/user/Users/az2798/ESM1b_data/"

LLR_dataframes = []
for protein_file in glob.glob(os.path.join(vault_dir, "*")):
    protein_id = os.path.basename(protein_file).split("_")[0]

    LLR_df = pd.read_csv(protein_file)
    LLR_df_melted = LLR_df.melt(id_vars='Unnamed: 0', var_name='Pos_Original_AA', value_name='LLR')
    LLR_df_melted.rename(columns={'Unnamed: 0': 'Changed_AA'}, inplace=True)

    LLR_df_melted[['Original_AA', 'Pos']] = LLR_df_melted['Pos_Original_AA'].str.extract(r'([A-Z])\s*(\d+)')
    LLR_df_melted = LLR_df_melted.dropna(subset=['Pos'])
    LLR_df_melted['Pos'] = LLR_df_melted['Pos'].astype(int)

    df_final = LLR_df_melted[['Pos', 'Original_AA', 'Changed_AA', 'LLR']]
    df_final["Protein_ID"] = protein_id
    LLR_dataframes.append(df_final)

final_ESM_df = pd.concat(LLR_dataframes, ignore_index=0, axis =0)
final_ESM_df.to_csv(os.path.join(
    output_dir, "All_LLR_scores.csv"
))
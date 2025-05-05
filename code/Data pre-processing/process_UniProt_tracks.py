import pandas as pd
import glob
import os


vault_dir = "/nfs/user/Users/az2798/UniProt_tracks/"
output_dir = "/home/az2798/MDmis/data/"

tracks_dataframes = []
for tracks_file in glob.glob(os.path.join(vault_dir, "*")):
    domain_type = os.path.basename(tracks_file).split("_")[-1].split(".")[0]

    track_df = pd.read_csv(tracks_file, sep = "\t", header = None)
    print(track_df.columns)
    track_df = track_df[[3, 13]]
    print(track_df[13])

    track_df[13] = track_df[13].str.partition(";")[0] #split by ; to get the <AA_LocStart-AA_LocEnd> code e.g M1-W9
    track_df[["Start", "End"]] = track_df[13].str.split('-', expand= True)
    track_df["Start"] = track_df["Start"].str[1:]
    track_df["End"] = track_df["End"].str[1:]

    track_df.rename(columns={3: "UniProtID", 13: "Domain_Start_End"}, inplace=True)

    track_df["Domain_Type"] = domain_type
    tracks_dataframes.append(track_df)

final_ESM_df = pd.concat(tracks_dataframes, ignore_index=0, axis =0)
final_ESM_df.to_csv(os.path.join(
    output_dir, "UniProt_tracks_processed.csv"
))
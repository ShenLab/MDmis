import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

import numpy as np

pd.set_option("display.max_columns", 100)
matplotlib.rcParams.update({'font.size': 13})

proteins_clinvar_mapped = pd.read_csv("/home/az2798/IDR_cons/data/amis_GERP_clinvar.csv", 
                                  index_col=0)
proteins_clinvar_mapped_all = proteins_clinvar_mapped[
    (proteins_clinvar_mapped["Data Source"] != "HGMD") &
    (proteins_clinvar_mapped["changed_residue"] != "X")
]

proteins_clinvar_mapped_all.loc[:,"MD Data Source_Binary"] = [
    "IDRs" if source == "IDRome" else
    "Other Protein Regions" for source in proteins_clinvar_mapped_all["MD Data Source"]
]

##########
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(proteins_clinvar_mapped_all[proteins_clinvar_mapped_all["MD Data Source_Binary"] == "IDRs"]["GERP++_RS"], bins=30, kde=False)
plt.xlabel("GERP++_RS")
plt.ylabel("Frequency")
plt.title("Histogram of GERP++_RS (IDRs)")
plt.grid(True)

plt.subplot(1, 2, 2)
sns.histplot(proteins_clinvar_mapped_all[proteins_clinvar_mapped_all["MD Data Source_Binary"] == "Other Protein Regions"]["GERP++_RS"], bins=30, kde=False)
plt.xlabel("GERP++_RS")
plt.ylabel("Frequency")
plt.title("Histogram of GERP++_RS (Other Protein Regions)")
plt.grid(True)

plt.tight_layout()
plt.savefig("/home/az2798/IDR_cons/results/GERP_RS_dist_protein_types.png", dpi=300, bbox_inches="tight")
plt.clf()
##########

plt.subplot(1, 2, 1)
sns.histplot(proteins_clinvar_mapped_all[(proteins_clinvar_mapped_all["outcome"] == 0) &
                                         (proteins_clinvar_mapped_all["MD Data Source_Binary"] == "IDRs")]["GERP++_RS"], bins=30, kde=False)
plt.xlabel("GERP++_RS")
plt.ylabel("Frequency")
plt.title("Histogram of GERP++_RS (Benign Variants)")
plt.grid(True)

plt.subplot(1, 2, 2)
sns.histplot(proteins_clinvar_mapped_all[(proteins_clinvar_mapped_all["outcome"] == 1) &
                                         (proteins_clinvar_mapped_all["MD Data Source_Binary"] == "IDRs")]["GERP++_RS"], bins=30, kde=False)
plt.xlabel("GERP++_RS")
plt.ylabel("Frequency")
plt.title("Histogram of GERP++_RS (Pathogenic Variants)")
plt.grid(True)

plt.tight_layout()
plt.savefig("/home/az2798/IDR_cons/results/GERP_RS_dist_variant_type.png", dpi=300, bbox_inches="tight")
plt.clf()
##########

# gerp_above_2 = proteins_clinvar_mapped_IDRome[proteins_clinvar_mapped_IDRome["GERP++_RS"] >= 2]
# percentage_above_2 = (len(gerp_above_2) / len(proteins_clinvar_mapped_IDRome)) * 100
# print(f"Percentage of GERP++_RS values >= 2: {percentage_above_2:.2f}%")

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(
    data = proteins_clinvar_mapped_all[proteins_clinvar_mapped_all["MD Data Source_Binary"] == "IDRs"],
      x="GERP++_RS", y="am_pathogenicity")
plt.xlabel("GERP++_RS")
plt.ylabel("AlphaMissense Score")
plt.title("GERP++_RS and AlphaMissense Probability (IDRs)")
plt.grid(True)

plt.subplot(1, 2, 2)
sns.kdeplot(
    data = proteins_clinvar_mapped_all[proteins_clinvar_mapped_all["MD Data Source_Binary"] == "Other Protein Regions"],
      x="GERP++_RS", y="am_pathogenicity")
plt.xlabel("GERP++_RS")
plt.ylabel("AlphaMissense Score")
plt.title("GERP++_RS and AlphaMissense Probability (Other Protein Regions)")
plt.grid(True)

plt.tight_layout()
plt.savefig("/home/az2798/IDR_cons/results/GERP_RS_AM_corr.png", dpi =300,
            bbox_inches = "tight")

plt.clf()

high_cons = proteins_clinvar_mapped_all[
    proteins_clinvar_mapped_all["GERP++_RS"] >= 2]
low_cons = proteins_clinvar_mapped_all[
    proteins_clinvar_mapped_all["GERP++_RS"] < 2]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(high_cons["outcome"], bins=2, kde=False)
plt.xlabel("Variant Effect")
plt.ylabel("Frequency")
plt.title("High Constraint - GERP++_RS >= 2")
plt.xticks([0.25, 0.75], ['Benign', 'Pathogenic'])
plt.grid(True)

# Plot for benign (GERP++_RS < 2)
plt.subplot(1, 2, 2)
sns.histplot(low_cons["outcome"], bins=2, kde=False)
plt.xlabel("Variant Effect")
plt.ylabel("Frequency")
plt.title("Low Constraint - GERP++_RS < 2")
plt.xticks([0.25, 0.75], ['Benign', 'Pathogenic'])
plt.grid(True)
plt.tight_layout()

plt.savefig("/home/az2798/IDR_cons/results/GERP_RS_labels.png", dpi =300,
            bbox_inches = "tight")


############


#First, we must convert all the 3 letter amino acid codes into 1 letter codes

pLDDT_scores = pd.read_csv("/home/az2798/IDR_cons/data/pLDDT_scores/processed_pLDDT_scores.csv",
                            index_col=0)

pLDDT_scores["location"] = pLDDT_scores["location"].astype(int)
proteins_clinvar_mapped_all["location"] = proteins_clinvar_mapped_all["location"].astype(int)

proteins_clinvar_pLDDT = pd.merge(left=proteins_clinvar_mapped_all,
                                  right=pLDDT_scores,
                                  left_on=["protein_id", "location"],
                                  right_on=["UniProtID", "location"],
                                  how = "left")


proteins_clinvar_pLDDT = proteins_clinvar_pLDDT.rename(columns={'outcome': 'Variant Effect'})
proteins_clinvar_pLDDT['Variant Effect'] = proteins_clinvar_pLDDT['Variant Effect'].map({0: 'Benign', 1: 'Pathogenic'})

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)



ax1= sns.kdeplot(
    data=proteins_clinvar_pLDDT[proteins_clinvar_pLDDT["MD Data Source_Binary"] == "IDRs"],
    x="GERP++_RS", y="pLDDT", hue="Variant Effect", 
    thresh=0.1, levels=5, linewidths=2, common_norm=False, legend=True
)


plt.xlabel("GERP++_RS")
plt.ylabel("pLDDT")
plt.title("GERP++_RS and AlphaFold pLDDT (IDRs)")
sns.move_legend(ax1, "upper left")
plt.grid(True)

plt.subplot(1, 2, 2)


ax2 = sns.kdeplot(
    data=proteins_clinvar_pLDDT[proteins_clinvar_pLDDT["MD Data Source_Binary"] == "Other Protein Regions"],
    x="GERP++_RS", y="pLDDT", hue="Variant Effect",
    thresh=0.1, levels=5, linewidths=2, common_norm=False, legend=True
)

plt.xlabel("GERP++_RS")
plt.ylabel("pLDDT")
plt.title("GERP++_RS and AlphaFold pLDDT (Other Protein Regions)")
sns.move_legend(ax2, "upper left")
plt.grid(True)

plt.tight_layout()
plt.savefig("/home/az2798/IDR_cons/results/GERP_RS_pLDDT_corr.png", dpi=300, bbox_inches="tight")
plt.show()

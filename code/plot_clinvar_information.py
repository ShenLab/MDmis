import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

pd.set_option("display.max_columns", 100)

results_dir = "/home/az2798/IDR_cons/results/"
clinvar_data = "/share/vault/Users/az2798/ClinVar_Data/training.csv"
data_dir = "/home/az2798/IDR_cons/data/"

df = pd.read_csv(clinvar_data)
mapped_protein_seqs_df = pd.read_csv(f"{data_dir}mapped_protein_seqs.csv",
                                    index_col=0)

df = df.rename(columns={'data_source': 'Data Source', 'score': 'Variant Effect'})

df['Variant Effect'] = df['Variant Effect'].map({0: 'Benign', 1: 'Pathogenic'})

proportions = df.groupby(['Data Source', 'Variant Effect']).size().unstack(fill_value=0)
proportions = proportions.div(proportions.sum(axis=1), axis=0)
fig = plt.figure()
spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 1], wspace= 0.25)
ax2 = fig.add_subplot(spec[0])
sns.histplot(
            y="Data Source",
            data=df,
            stat="count", shrink=.7, ax=ax2)
ax3 = fig.add_subplot(spec[1], sharey=ax2)
sns.histplot(
            y="Data Source",
            hue="Variant Effect",
            data=df,
            stat="count",
            multiple="fill", shrink=.7, ax=ax3)


for (i, p) in enumerate(ax2.patches):

    ax2.annotate(f'n={p.get_width():.0f}', 
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', va='center', xytext=(8, 0), textcoords='offset points', fontsize=8)
ax3.legend(title = "Variant Effect", labels = ["Pathogenic", "Benign"],
           loc='upper center', bbox_to_anchor=(0.5, -0.1))
ax2.tick_params(labelleft=True)
ax3.set_ylabel("")
ax3.set_xlabel("Proportion")
ax3.tick_params(labelleft=False)
ax2.set_ylabel("Data Source")
ax2.set_xlabel("Number of Samples")
fig.savefig(f'{results_dir}clinvar_info.png', dpi =300,bbox_inches = "tight")

plt.clf()

#######


clinvar_proteins = pd.merge(left = mapped_protein_seqs_df,
                          right = df, left_on="UniProtID", right_on = "uniprotID", how = "inner")
clinvar_proteins
clinvar_proteins.drop_duplicates(subset = ["UniProtID"], inplace=True)
clinvar_proteins = clinvar_proteins.rename(columns={'source': 'MD Data Source'})

proportions = clinvar_proteins.groupby(['MD Data Source', 'Variant Effect']).size().unstack(fill_value=0)
proportions = proportions.div(proportions.sum(axis=1), axis=0)
fig = plt.figure()
spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 1], wspace= 0.25)
ax2 = fig.add_subplot(spec[0])
sns.histplot(
            y="MD Data Source",
            data=clinvar_proteins,
            stat="count", shrink=.7, ax=ax2)
ax3 = fig.add_subplot(spec[1], sharey=ax2)
sns.histplot(
            y="MD Data Source",
            hue="Variant Effect",
            data=clinvar_proteins,
            stat="count",
            multiple="fill", shrink=.7, ax=ax3)


for (i, p) in enumerate(ax2.patches):

    ax2.annotate(f'n={p.get_width():.0f}', 
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', va='center', xytext=(8, 0), textcoords='offset points', fontsize=8)
ax3.legend(title = "Variant Effect", labels = ["Pathogenic", "Benign"],
           loc='upper center', bbox_to_anchor=(0.5, -0.1))
ax2.tick_params(labelleft=True)
ax3.set_ylabel("")
ax3.set_xlabel("Proportion")
ax3.tick_params(labelleft=False)
ax2.set_ylabel("MD Data Source")
ax2.set_xlabel("Number of Samples")
fig.savefig(f'{results_dir}clinvar_md_info.png', dpi =300,bbox_inches = "tight")

plt.clf()
#######

pLI_data = pd.read_csv(f'{data_dir}pLI_data/pliByTranscript.bed', delimiter = "\t",
                       names=["chrom", "chromStart", "chromEnd", "name", "score", "strand",
                              "thickStart", "thickEnd", "itemRgb", "blockCount",
                              "blockSizes", "chromStarts", "_mouseOver", "_loeuf", "_pli",
                              "geneName", "synonymous", "pLoF"],
                              index_col = False)

print(pLI_data)

pLI_by_mapped_proteins = pd.merge(left = pLI_data, right = clinvar_proteins, left_on = "name",
                                  right_on = "ENST", how = "inner")

print(pLI_by_mapped_proteins.shape)

ax = sns.violinplot(x="MD Data Source", y="_pli", hue = "Variant Effect",
                    data = pLI_by_mapped_proteins)

ax.set_ylabel("pLI")
ax.figure.savefig(f'{results_dir}clinvar_md_constrained_genes.png', dpi =300,bbox_inches = "tight")


 

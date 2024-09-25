import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import numpy as np
from MDmis_RF import create_feature_table, load_h5py_data

pd.set_option("display.max_columns", 100)
matplotlib.rcParams.update({'font.size': 13})



def main():

    h5py_path = "/share/vault/Users/az2798/train_data_all/filtered_feature_all_ATLAS_GPCRmd_IDRome.h5"
    proteins_clinvar_mapped = pd.read_csv("/home/az2798/IDR_cons/data/amis_GERP_clinvar.csv", 
                                    index_col=0)

    proteins_clinvar_mapped_all = proteins_clinvar_mapped[
        (proteins_clinvar_mapped["Data Source"] != "HGMD") &
        (proteins_clinvar_mapped["changed_residue"] != "X")
    ]

    proteins_clinvar_mapped_IDRome = proteins_clinvar_mapped_all[
        proteins_clinvar_mapped_all["MD Data Source"] == "IDRome"]

    #proteins_clinvar_mapped_others = proteins_clinvar_mapped_all[
    #    proteins_clinvar_mapped_all["MD Data Source"] != "IDRome"]
    
    aaindex_res_mat = np.load("/home/az2798/IDR_cons/data/aa_index1.npy")
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    #dssp_column_names = ['Beta Bridge', 'Extended Strand',
    #                    'Alpha Helix', '5-helix', '3-helix',
    #                    'Bend', 'Hydrogen Bonded Turn', 'Loops and Irregular Elements']
    dssp_column_names = ['Turn/Bend', 'Hydrogen Bonded Turn', 'Loops and Irregular Elements']
    dssp_md_columns = ['Res_MD_9', 'Res_MD_10', 'Res_MD_11']
    res_data, pair_data = load_h5py_data(h5py_path)

    print("check 1, Loaded h5py")
    IDRs_feature_table = create_feature_table(proteins_clinvar_mapped_IDRome, res_data, pair_data, 
                         aaindex_res_mat, aa_order, use_res_md=True, use_pair_md=True)

    #others_feature_table = create_feature_table(proteins_clinvar_mapped_others, res_data, pair_data, 
    #                     aaindex_res_mat, aa_order, use_res_md=True, use_pair_md=True)
    
    IDRs_feature_table.to_csv("/home/az2798/IDR_cons/data/RF_train_val/IDRs_feature_table.csv")
    #others_feature_table.to_csv("/home/az2798/IDR_cons/data/RF_train_val/others_feature_table.csv")


    IDRs_feature_table = IDRs_feature_table.rename(columns={'outcome': 'Variant Effect'})
    IDRs_feature_table['Variant Effect'] = IDRs_feature_table['Variant Effect'].map({0: 'Benign', 1: 'Pathogenic'})


    IDRs_res_features = IDRs_feature_table[dssp_md_columns + ["Variant Effect"]].melt(id_vars = "Variant Effect",
                                                                var_name="DSSP", value_name = "Proportion")
    #IDRs_res_features_pathogenic = IDRs_feature_table[
    #    IDRs_feature_table["outcome"] == 1][dssp_md_columns].melt(var_name='DSSP', value_name='Proportion')

    #IDRs_res_features_benign = IDRs_feature_table[
    #    IDRs_feature_table["outcome"] == 0][dssp_md_columns].melt(var_name='DSSP', value_name='Proportion')


    #print(IDRs_res_features_pathogenic.shape, "Pathogenic shape")
    #print(IDRs_res_features_pathogenic.head(20))

    #print(IDRs_res_features_benign.shape, "Benign shape")
    #print(IDRs_res_features_benign.head(20))

    #others_res_features_high_cons = others_feature_table[
    #    others_feature_table["GERP_RS_score"] > 2.0][dssp_md_columns].melt(var_name='DSSP', value_name='Proportion')

    #others_res_features_low_cons = others_feature_table[
    #    others_feature_table["GERP_RS_score"] <= 2.0][dssp_md_columns].melt(var_name='DSSP', value_name='Proportion')

    ##########
    plt.figure(figsize=(10, 6))
    sns.barplot(data=IDRs_res_features, x='DSSP', y='Proportion',
                hue='Variant Effect', ci=68)
    plt.xlabel("DSSP")
    plt.ylabel("Proportion of DSSP Assignment")
    plt.title("Secondary Structure and Variant Pathogenicity (IDRs)")
    plt.xticks(ticks = np.arange(len(dssp_column_names)), 
               labels = dssp_column_names)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("/home/az2798/IDR_cons/results/GERP_variant_type_DSSP.png", dpi=300, bbox_inches="tight")
    plt.clf()
    
    print("check 2, Plotted")

    ##########
if __name__ == "__main__":
    main()
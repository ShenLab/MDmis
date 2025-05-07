# coding=utf-8
import math
import random
import sys
import warnings

import mdtraj
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)

pd.set_option("display.max_columns", None)


def get_seq_from_pdb(pdb_file_path):
    with open(pdb_file_path, "r") as pdb_file:
        for record in SeqIO.parse(pdb_file, "pdb-atom"):
            sequence = record.seq
            return str(sequence)


def get_dssp_sasa_chi1_rmsf_cov(traj, seq, ca_atoms, N, n_frames):
    """
    Computes and returns various residue-level features from a molecular dynamics trajectory.

    This function calculates the following features for each residue:
    - DSSP (secondary structure assignments)
    - SASA (solvent accessible surface area) mean and standard deviation
    - Dihedral angles (chi1, phi, psi) histogram counts
    - RMSF (root mean square fluctuation)
    - Covariance of CA atoms

    Args:
        traj (mdtraj.Trajectory): MDTraj trajectory object containing the molecular dynamics trajectory.
        seq (str): The sequence of the protein.
        ca_atoms (np.ndarray): Array of indices of CA atoms in the trajectory.
        N (int): Number of CA atoms.
        n_frames (int): Number of frames in the trajectory.

    Returns:
        pd.DataFrame: DataFrame containing the computed residue-level features.
        np.ndarray: Covariance matrix of CA atoms.
    """
    # Compute DSSP (secondary structure assignments)
    dssp = mdtraj.compute_dssp(traj, simplified=False)
    dssp = pd.DataFrame(dssp)
    dssp_count = pd.DataFrame(
        [dssp[i].value_counts() for i in dssp.columns],
        columns=["B", "E", "H", "I", "G", "S", "T", " "],
        index=dssp.columns,
    )
    dssp_count.columns = ["dssp_" + i for i in dssp_count.columns]

    # Define the residue level scores matrix
    res_score = dssp_count.fillna(0) / n_frames
    res_score.insert(0, "AA", list(seq))
    res_score.insert(1, "ca_atom", ca_atoms)

    # Compute SASA (solvent accessible surface area)
    sasa = mdtraj.shrake_rupley(traj, mode="residue")
    res_score["sasa_mean"] = sasa.mean(axis=0)
    res_score["sasa_std"] = sasa.std(axis=0)

    # Compute dihedral angles (chi1, phi, psi)
    angle_bins = [math.pi * (-1 + i / 6) for i in range(0, 13)]
    chi1 = mdtraj.compute_chi1(traj)
    phi = mdtraj.compute_phi(traj)
    psi = mdtraj.compute_psi(traj)
    angles = {"chi": chi1, "phi": phi, "psi": psi}
    for angle in angles:
        angle_count = []
        for i in range(angles[angle][1].shape[1]):
            counts, bins = np.histogram(angles[angle][1][:, i], bins=angle_bins)
            angle_count.append(counts)

        angle_tab = pd.DataFrame(
            np.array(angle_count), columns=[f"{angle}_{k}" for k in range(12)]
        )
        if angle == "phi":
            angle_tab.index = angles[angle][0][:, 2]
        else:
            angle_tab.index = angles[angle][0][:, 1]
        angle_tab = angle_tab / n_frames

        res_score = pd.merge(
            res_score, angle_tab, left_on="ca_atom", right_index=True, how="left"
        )

    # Compute RMSF (root mean square fluctuation)
    res_score["rmsf"] = mdtraj.rmsf(traj, traj, atom_indices=ca_atoms)

    # Compute covariance matrix of CA atoms
    ca_xyz = traj.xyz[:, ca_atoms, :]
    cov_3n = np.cov(ca_xyz.reshape((-1, len(ca_atoms) * 3)), rowvar=False)
    cov_n = np.trace(cov_3n.reshape((N, 3, N, 3)), axis1=1, axis2=3)

    return res_score, cov_n


def get_nparray_from_df(res_df, col_angle, col_move, col_drop):
    """
    Processes the DataFrame to fill NaNs in angle columns and create a numpy array with selected columns.

    Args:
        res_df (pd.DataFrame): The DataFrame containing residue features.
        col_angle (list): List of angle-related column names.
        col_move (list): List of column names to be moved to the front.
        col_drop (list): List of column names to be dropped.

    Returns:
        np.ndarray: The resulting numpy array.
    """
    # Fill NaN values in angle-related columns with 1/12
    for col in col_angle:
        if col in res_df.columns:
            res_df[col] = res_df[col].fillna(1 / 12)

    # Create a list of columns to include in the numpy array
    # Essentially move the sasa_mean, sasa_std, and rmsf first in the list follower by the angle columns
    included_cols = col_move + [
        col for col in res_df.columns if col not in col_move + col_drop
    ]
    #print("Columns", included_cols)
    # Convert the DataFrame to a numpy array, specifying the included columns
    #print("Df with columns", res_df[included_cols])
    res_array = res_df[included_cols].to_numpy(dtype=np.float16)
    return res_array

def calcRg(t,residues,seq):
    ##CREDIT goes to Tesei 2023's Github 
    ##Copied code for replicability
    ##https://github.com/KULL-Centre/_2023_Tesei_IDRome/tree/main

    fasta = list(seq)
    masses = residues.loc[fasta,'MW'].values
    # calculate the center of mass
    cm = np.sum(t.xyz*masses[np.newaxis,:,np.newaxis],axis=1)/masses.sum()
    # calculate residue-cm distances
    si = np.linalg.norm(t.xyz - cm[:,np.newaxis,:],axis=2)
    # calculate rg
    rgarray = np.sqrt(np.sum(si**2*masses,axis=1)/masses.sum())
    return rgarray

def calcRs(traj):
    pairs = traj.top.select_pairs('all','all')
    d = mdtraj.compute_distances(traj,pairs)
    nres = traj.n_atoms
    ij = np.arange(2,nres,1)
    diff = [x[1]-x[0] for x in pairs]
    dij = np.empty(0)
    for i in ij:
        dij = np.append(dij, np.sqrt((d[:,diff==i]**2).mean().mean()))
    return ij,dij,np.mean(1/d,axis=1)

def compute_conformational_properties(traj_file,
    pdb_file, residues_masses,
    conf_properties_path, quantile,
    num_sample_frames):
    if ".xtc" in traj_file:
        traj = mdtraj.load_xtc(traj_file, pdb_file)
    elif ".dcd" in traj_file:
        traj = mdtraj.load_dcd(traj_file, pdb_file)
    else:
        raise ValueError(f"{traj_file} is of unsupported format.")
    total_frames = traj.n_frames
    frame_threshold_mdtraj = int(total_frames * quantile)
    sampled_frames = random.sample(range(frame_threshold_mdtraj, total_frames), k = num_sample_frames)
    final_traj = traj[sampled_frames]  # sample frames by burning first threshold number and picking from the remainder

    print("Loaded trajectory information", final_traj)
    ca_atoms = final_traj.topology.select("name CA")
    final_traj.superpose(final_traj, 0, atom_indices=ca_atoms)
    N = len(ca_atoms)
    sequence = get_seq_from_pdb(pdb_file)

    Rg = np.mean(calcRg(final_traj,residues=residues_masses, seq=sequence))
    ete = np.mean(mdtraj.compute_distances(final_traj,atom_pairs=[[0,N-1]]).flatten())

    #nonlinear scaling exponent, nu
    def scaling_exp(x, R0, v):
        return R0*np.power(x,v) 
    
    
    ij,dij,invrij = calcRs(final_traj)

    popt, pcov = curve_fit(scaling_exp, ij[ij>5],dij[ij>5],p0=[.4,.5] )
    nu = popt[1]
    print(nu)
    #print(Rg)
    #print(ete)

    np.save(conf_properties_path, arr=np.array([Rg, ete, nu]))

    
def process_md_data(
    traj_file,
    pdb_file,
    contact_path,
    res_feature_path,
    pair_feature_path,
    quantile,
    num_sample_frames,
):
    if ".xtc" in traj_file:
        traj = mdtraj.load_xtc(traj_file, pdb_file)
    elif ".dcd" in traj_file:
        traj = mdtraj.load_dcd(traj_file, pdb_file)
    else:
        raise ValueError(f"{traj_file} is of unsupported format.")
    total_frames = traj.n_frames
    frame_threshold_mdtraj = int(total_frames * quantile)
    
    
    #traj = traj[frame_threshold_mdtraj:]

    sampled_frames = random.sample(range(frame_threshold_mdtraj, total_frames), k = num_sample_frames)
    final_traj = traj[sampled_frames]  # sample frames by burning first threshold number and picking from the remainder

    print("Loaded trajectory information", final_traj)
    assert final_traj.n_frames == num_sample_frames, f"Number of frames post processing {final_traj.n_frames} not equal to provided number {num_sample_frames}."
    ca_atoms = final_traj.topology.select("name CA")
    final_traj.superpose(final_traj, 0, atom_indices=ca_atoms)
    N = len(ca_atoms)

    sequence = get_seq_from_pdb(pdb_file)
    res_feat, cov_n = get_dssp_sasa_chi1_rmsf_cov(
        final_traj, sequence, ca_atoms, N, num_sample_frames
    )

    contacts_df = pd.read_csv(
        contact_path,
        skiprows=2,
        sep="\t",
        header=None,
        usecols=range(4),
        names=["frame", "interaction", "atom1", "atom2"],
    )
    
    #frame_threshold_contacts = contacts_df["frame"].quantile(quantile)
    #contacts_df = contacts_df[contacts_df["frame"] >= frame_threshold_contacts]
    
    contacts_df = contacts_df[contacts_df["frame"].isin(sampled_frames)]
    print(contacts_df)
    
    res1_nums = contacts_df["atom1"].str.split(":", expand=True)[2].astype(int)
    res2_nums = contacts_df["atom2"].str.split(":", expand=True)[2].astype(int)
    min_res_num = min(res1_nums.min(), res2_nums.min())

    contacts_df["res1"] = res1_nums - min_res_num  # pushing everything to be 0 indexed
    contacts_df["res2"] = res2_nums - min_res_num

    category_mapping = {
        "vdw": 0,
        "hbbb": 1,
        "hbsb": 2,
        "hbss": 3,
        "hp": 4,
        "sb": 5,
        "pc": 6,
        "ps": 7,
        "ts": 8,
    }
    contacts_df["interaction_id"] = contacts_df["interaction"].map(category_mapping)
    contacts_df = contacts_df.dropna(subset=["interaction_id"], axis=0)
    contacts_df["interaction_id"] = contacts_df["interaction_id"].astype(int)

    #print("Contacts Df", contacts_df)
    contacts_count = (
        contacts_df.groupby(["res1", "res2", "interaction_id"]).count()[["frame"]]
        / num_sample_frames
    )
    #print("Contacts Count", contacts_count)
    contact_map = np.zeros([N, N, 9])
    for i in contacts_count.index:
        res1, res2, interaction_id = i
        contact_map[res1, res2, interaction_id] = contacts_count.loc[i]

    pair_feat = np.concatenate([contact_map, cov_n.reshape(N, N, 1)], axis=2).astype(
        np.float16
    )
    print("Pair feature array shape", pair_feat.shape)
    np.save(pair_feature_path, arr=pair_feat)

    col_angle = [f"{angle}_{i}" for angle in ["chi", "phi", "psi"] for i in range(12)]
    col_move = ["sasa_mean", "sasa_std", "rmsf"]
    col_drop = ["AA", "ca_atom"]

    res_array = get_nparray_from_df(res_feat, col_angle, col_move, col_drop)
    print("Res feature array shape", res_array.shape)
    np.save(res_feature_path, arr=res_array)


def main():
    #example run
    MD_dir = "/path/to/MD" # change as needed for testing
    process_md_data(
        f"{MD_dir}aa_traj.dcd",
        f"{MD_dir}aa_traj.pdb",
        f"{MD_dir}contacts.tsv",
        f"{MD_dir}res_feature.npy",
        f"{MD_dir}pair_feature.npy",
        0.2,
        500,
    )

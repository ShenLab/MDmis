import numpy as np
import os
import shutil
import itertools
import numpy as np
import pandas as pd
import scipy.stats as scs
from scipy.optimize import curve_fit
from sklearn.covariance import LedoitWolf
from scipy import constants
from numpy import linalg
from localcider.sequenceParameters import SequenceParameters
import mdtraj as md
from simtk import openmm, unit
from simtk.openmm import app
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def genParamsLJ(df,seq,Nc,Cc):
    fasta = seq.copy()
    r = df.copy()
    if Nc == 1:
        r.loc['X'] = r.loc[fasta[0]]
        r.loc['X','MW'] += 2
    if Cc == 1:
        r.loc['Z'] = r.loc[fasta[-1]]
        r.loc['Z','MW'] += 16
    lj_eps = 0.2*4.184
    lj_sigma = pd.DataFrame((r.sigmas.values+r.sigmas.values.reshape(-1,1))/2,
                            index=r.sigmas.index,columns=r.sigmas.index)
    lj_lambda = pd.DataFrame((r.lambdas.values+r.lambdas.values.reshape(-1,1))/2,
                             index=r.lambdas.index,columns=r.lambdas.index)
    return lj_eps, lj_sigma, lj_lambda

def genParamsDH(df,seq,temp,ionic,Nc,Cc,Hc):
    kT = 8.3145*temp*1e-3
    fasta = seq.copy()
    r = df.copy()
    # Set the charge on HIS based on the pH of the protein solution
    r.loc['H','q'] = Hc
    if Nc == 1:
        r.loc['X'] = r.loc[fasta[0]]
        r.loc['X','q'] = r.loc[seq[0],'q'] + 1.
        fasta[0] = 'X'
    if Cc == 1:
        r.loc['Z'] = r.loc[fasta[-1]]
        r.loc['Z','q'] = r.loc[seq[-1],'q'] - 1.
        fasta[-1] = 'Z'
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    yukawa_eps = [r.loc[a].q*np.sqrt(lB*kT) for a in fasta]
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.022/10)
    return yukawa_eps, yukawa_kappa

def genDCD(name, seq, results_dir, eqsteps=10):
    """
    Generates coordinate and trajectory
    in convenient formats
    """
    traj = md.load(os.path.join(results_dir,
        "{:s}/pretraj.dcd".format(name)),
        top=os.path.join(results_dir, "{:s}/top.pdb".format(name))
        )
    cgtop = md.Topology()
    cgchain = cgtop.add_chain()
    for aa in seq:
        cgres = cgtop.add_residue(aa, cgchain)
        cgtop.add_atom('CA', element=md.element.carbon, residue=cgres)
    for i in range(traj.n_atoms-1):
        cgtop.add_bond(cgtop.atom(i),cgtop.atom(i+1))
    traj = md.Trajectory(traj.xyz, cgtop, traj.time, traj.unitcell_lengths, traj.unitcell_angles)
    traj = traj.image_molecules(inplace=False, anchor_molecules=[set(traj.top.chain(0).atoms)], make_whole=True)
    traj.center_coordinates()
    traj.xyz += traj.unitcell_lengths[0,0]/2
    traj[int(eqsteps):].save_dcd(os.path.join(results_dir,
                                              "{:s}/traj.dcd".format(name))
    )
    traj[int(eqsteps)].save_pdb(os.path.join(results_dir,
                                             "{:s}/top.pdb".format(name))
    )

def simulate(name,seq,residues, results_dir,
        temp,ionic,Nc,Cc,Hc,nsteps,stride=1e3,eqsteps=1000):
    
    os.makedirs(
        os.path.join(results_dir, 
                     name),
                     exist_ok = True
    )

    lj_eps, _, _ = genParamsLJ(residues,seq,Nc,Cc)
    yukawa_eps, yukawa_kappa = genParamsDH(residues,seq,temp,ionic,Nc,Cc,Hc)

    N = len(seq)
    L = (N-1)*0.38+4

    system = openmm.System()

    # set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = L * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = L * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = L * unit.nanometers
    system.setDefaultPeriodicBoxVectors(a, b, c)

    top = md.Topology()
    pos = []
    chain = top.add_chain()
    pos.append([[0,0,L/2+(i-N/2.)*.38] for i in range(N)])
    for resname in seq:
        residue = top.add_residue(resname, chain)
        top.add_atom(resname, element=md.element.carbon, residue=residue)
    for i in range(chain.n_atoms-1):
        top.add_bond(chain.atom(i),chain.atom(i+1))
    md.Trajectory(np.array(pos).reshape(N,3), top, 0, [L,L,L], [90,90,90]).save_pdb(
        os.path.join(results_dir,
                     '{:s}/top.pdb'.format(name))
    )

    pdb = app.pdbfile.PDBFile(
        os.path.join(results_dir, '{:s}/top.pdb'.format(name))
    )

    system.addParticle((residues.loc[seq[0]].MW+2)*unit.amu)
    for a in seq[1:-1]:
        system.addParticle(residues.loc[a].MW*unit.amu)
    system.addParticle((residues.loc[seq[-1]].MW+16)*unit.amu)

    hb = openmm.openmm.HarmonicBondForce()
    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    ah = openmm.openmm.CustomNonbondedForce(energy_expression+'; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/2)^12-(0.5*(s1+s2)/2)^6')
    yu = openmm.openmm.CustomNonbondedForce('q*(exp(-kappa*r)/r - exp(-kappa*4)/4); q=q1*q2')
    yu.addGlobalParameter('kappa',yukawa_kappa/unit.nanometer)
    yu.addPerParticleParameter('q')

    ah.addGlobalParameter('eps',lj_eps*unit.kilojoules_per_mole)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')

    for a,e in zip(seq,yukawa_eps):
        yu.addParticle([e*unit.nanometer*unit.kilojoules_per_mole])
        ah.addParticle([residues.loc[a].sigmas*unit.nanometer, residues.loc[a].lambdas*unit.dimensionless])

    for i in range(N-1):
        hb.addBond(i, i+1, 0.38*unit.nanometer, 8033*unit.kilojoules_per_mole/(unit.nanometer**2))
        yu.addExclusion(i, i+1)
        ah.addExclusion(i, i+1)

    yu.setForceGroup(0)
    ah.setForceGroup(1)
    yu.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    hb.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4*unit.nanometer)
    ah.setCutoffDistance(2*unit.nanometer)

    system.addForce(hb)
    system.addForce(yu)
    system.addForce(ah)

    #serialized_system = XmlSerializer.serialize(system)
    #outfile = open('system.xml','w')
    #outfile.write(serialized_system)
    #outfile.close()

    integrator = openmm.openmm.LangevinIntegrator(temp*unit.kelvin,0.01/unit.picosecond,0.010*unit.picosecond) #10 fs timestep

    platform = openmm.Platform.getPlatformByName('CUDA')

    simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform, dict(CudaPrecision='mixed'))

    check_point = os.path.join(results_dir,
                               '{:s}/restart.chk'.format(name)
    )

    if os.path.isfile(check_point):
        print('Reading check point file')
        simulation.loadCheckpoint(check_point)
        simulation.reporters.append(app.dcdreporter.DCDReporter(
            os.path.join(results_dir,'{:s}/pretraj.dcd'.format(name)),int(stride),append=True))
    else:
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(app.dcdreporter.DCDReporter(
            os.path.join(results_dir, '{:s}/pretraj.dcd'.format(name)),int(stride)))

    simulation.reporters.append(app.statedatareporter.StateDataReporter(
        os.path.join(results_dir, '{:s}/traj.log'.format(name)),int(stride),
             potentialEnergy=True,temperature=True,step=True,speed=True,elapsedTime=True,separator='\t'))

    simulation.step(nsteps)

    simulation.saveCheckpoint(check_point)

    genDCD(name,[residues.three[aa] for aa in seq],
           results_dir, eqsteps)


def run_md_sim(NAME, 
               SEQUENCE,
               residues_file,
               results_dir,
               charged_N_terminal_amine=True,
               charged_C_terminal_carboxyl = False,
               charged_histidine = False, 
	       Simulation_time = "AUTO"):
    
    
    # NAME = "Q99457_1_97" #@param {type:"string"}

    # SEQUENCE = "MAEADFKMVSEPVAHGVAEEEMASSTSDSGEESDSSSSSSSTSDSSSSSSTSGSSSGSGSSSSSSGSTSSRSRLYRKKRVPEPSRRARRAPLGTNFV" #@param {type:"string"}
    if " " in SEQUENCE:
        SEQUENCE = ''.join(SEQUENCE.split())
        print('Blank character(s) found in the provided sequence. Sequence has been corrected, but check for integrity:')
        print(SEQUENCE)
        print('\n')
    Temperature = 310
    Ionic_strength = 0.150 

    Nc = 1 if charged_N_terminal_amine == True else 0
    Cc = 1 if charged_C_terminal_carboxyl == True else 0

    if charged_histidine == True:
        print('Define pH and pKa to set the charge of Histidines according to the Henderson-Hasselbalch equation.')
        pH = input('Enter pH value: ')
        pH = float(pH)
        pKa = input('Enter pKa value: ')
        pKa = float(pKa)
        Hc = 1/(1+10**(pH-pKa))
    if charged_histidine == False:
        Hc = 0

    residues = pd.read_csv(residues_file)
    residues = residues.set_index('one')

    N_res = len(SEQUENCE)
    N_save = 7000 if N_res < 150 else int(np.ceil(3e-4*N_res**2)*1000)

    if Simulation_time == "AUTO":
        nsteps = 1010*N_save
        print('AUTO simulation length selected. Running for {} ns'.format(nsteps*0.01/1000))
    else:
        nsteps = float(Simulation_time)*1000/0.01//N_save*N_save
        print('Simulation time selected. Running for {} ns'.format(nsteps*0.01/1000))
    try:
        shutil.rmtree(NAME)
    except:
        pass
    simulate(NAME,list(SEQUENCE),residues, results_dir, 
             temp=Temperature,ionic=Ionic_strength,Nc=Nc,Cc=Cc,
            Hc=Hc,nsteps=nsteps,stride=N_save,eqsteps=10)
    



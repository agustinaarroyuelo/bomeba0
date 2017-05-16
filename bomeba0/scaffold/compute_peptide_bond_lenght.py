"""Use PyMOl to create templates for bomeba"""

import numpy as np
np.set_printoptions(precision=3)
import __main__
__main__.pymol_argv = ['pymol','-qck']
import pymol
from pymol import cmd, stored
pymol.finish_launching()
import openbabel as ob

def minimize(selection='all', forcefield='MMFF94s', method='cg',
             nsteps= 2000, conv=1E-6, cutoff=False, cut_vdw=6.0, cut_elec=8.0):
    pdb_string = cmd.get_pdbstr(selection)
    name = cmd.get_legal_name(selection)
    obconversion = ob.OBConversion()
    obconversion.SetInAndOutFormats('pdb', 'pdb')
    mol = ob.OBMol()
    obconversion.ReadString(mol, pdb_string)
    ff = ob.OBForceField.FindForceField(forcefield)
    ff.Setup(mol)
    if cutoff == True:
        ff.EnableCutOff(True)
        ff.SetVDWCutOff(cut_vdw)
        ff.SetElectrostaticCutOff(cut_elec)
    if method == 'cg':
        ff.ConjugateGradients(nsteps, conv)
    else:
        ff.SteepestDescent(nsteps, conv)
    ff.GetCoordinates(mol)
    nrg = ff.Energy()
    pdb_string = obconversion.WriteString(mol)
    cmd.delete(name)
    if name == 'all':
        name = 'all_'
    cmd.read_pdbstr(pdb_string, name)
    return nrg


pbl = []
for res_name_i in aa:
    for res_name_j in aa:
        cmd.fab(res_name_i + res_name_j)
        minimize(selection=sel, forcefield='GAFF')
        a = cmd.get_distance('resi 1 and name C', 'resi 2 and name N')
        pbl.append(a)
        cmd.delete('all')
mean = sum(pbl) / len(pbl)
print(mean)


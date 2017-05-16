"""Use PyMOl to create templates for bomeba"""

import numpy as np
np.set_printoptions(precision=3)
import __main__
__main__.pymol_argv = ['pymol','-qck']
import pymol
from pymol import cmd, stored
pymol.finish_launching()
import openbabel as ob

# set hydrogen names to PDB compliant
cmd.set('pdb_reformat_names_mode', 1)

sel = 'all'

aa = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M',
      'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def minimize(selection='all', forcefield='MMFF94s', method='cg',
             nsteps= 2000, conv=1E-8, cutoff=False, cut_vdw=6.0, cut_elec=8.0):
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

#aa = ['A']

for res_name in aa:
    ## Get coordinates and offset
    cmd.fab(res_name)
    nrg = minimize(selection=sel, forcefield='GAFF', method='cg', nsteps=2000)
    #print(nrg)
    xyz = cmd.get_coords(sel)
    offset = len(xyz)

    ## get atom names
    stored.atom_names = []
    cmd.iterate(sel, 'stored.atom_names.append(name)')

    ## get bonds
    stored.bonds = []
    model = cmd.get_model(sel)
    for at in model.atom:
        cmd.iterate('neighbor ID %s' % at.id, 
                        'stored.bonds.append((%s-1, ID-1))' % at.id)
    bonds = list(set([tuple(sorted(i)) for i in stored.bonds]))
    bonds.sort()
    
    ## get bb and sc
    stored.bb = []
    stored.sc = []
    bb = '(nbr (name n+ca+c+o) ) and hydro or (name n+ca+c+o)'
    sc = 'not ({})'.format(bb)
    cmd.iterate(bb, 'stored.bb.append(ID-1)')
    cmd.iterate(sc, 'stored.sc.append(ID-1)')
    
    ## small check before returning the results
    if len(stored.atom_names) == offset:
    
        if res_name == 'G':
            stored.atom_names = ['N', 'CA', 'C', 'O', 'H', 'HA2', 'HA3']
        res = """{}_info = AA_info(coords=np.{},
             atom_names = {},
             bb = {},
             sc = {},
             bonds = {},
             offset = {})\n""".format(res_name, repr(xyz), stored.atom_names, stored.bb, stored.sc, bonds, offset)
        print(res)
    else:
        print('Something funny is going on here!')
    cmd.delete('all')



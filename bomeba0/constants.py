"""
A collection of common constants and forcefield parameters
"""
from collections import namedtuple
Constants = namedtuple('Constants', [
                       'radians_to_degrees', 'degrees_to_radians', 'pi', 'peptide_bond_lenght'])
constants = Constants(
    57.29577951308232, 0.017453292519943295, 3.141592653589793, 1.32)

#par_vdw = {'H':(1.49, 0.015), 'C':(1.90, 0.086), 'O':(1.66, 0.210), 'N':(1.82, 0.170)}
# Using par_vdw I create a dictionary with precomputed sigma_ij
#sigma_ij = par_vdw[key_i][0] + par_vdw[key_j][0]
par_s_ij = {'CH': 3.39,
            'HN': 3.31,
            'OH': 3.15,
            'NO': 3.48,
            'HO': 3.15,
            'OC': 3.56,
            'CN': 3.72,
            'OO': 3.32,
            'HC': 3.39,
            'NC': 3.72,
            'NH': 3.31,
            'NN': 3.64,
            'CC': 3.8,
            'ON': 3.48,
            'HH': 2.98,
            'CO': 3.56}
# Using par_vdw I create a dictionary with precomputed epsilon_ij
#epsilon_ij = (par_vdw[key_i][1] * par_vdw[key_j][1])**0.5
par_eps_ij = {'CH': 0.03591656999213594, 
              'HN': 0.05049752469181039,
              'OH': 0.05612486080160912,
              'NO': 0.18894443627691185,
              'HO': 0.05612486080160912,
              'OC': 0.1343874994186587,
              'CN': 0.12091319200153472,
              'OO': 0.21,
              'HC': 0.03591656999213594,
              'NC': 0.12091319200153472,
              'NH': 0.05049752469181039,
              'NN': 0.17,
              'CC': 0.086,
              'ON': 0.18894443627691185,
              'HH': 0.015,
              'CO': 0.1343874994186587}

###############################################################################
##### below this line we are not using anything, just here for the record #####
###############################################################################
"""
C: carbonyl carbon
CR: carbon with no hydrogens,
CR1E: extended aromatic carbon with 1 H
CH1E: extended aliphatic carbon with 1 H
CH2E: extended aliphatic carbon with 2 H
CH3E: extended aliphatic carbon with 3 H
NH1: amide nitrogen
NR: aromatic nitrogen with no hydrogens
NH2: nitrogen bound to two hydrogens
NH3: nitrogen bound to three hydrogens
NC2: guanidinium nitrogen
N: proline nitrogen
OH1: hydroxyl oxygen
O: carbonyl oxygen
OC: carboxyl oxygen
S: sulphur
SH1E: extended sulphur with one hydrogen
H?: this entry seems to be an hydrogen
H?: this entry seems to be an hydrogen

see table I Effective Energy Function for Proteins in Solution 
Themis Lazaridis and Martin Karplus

"""

# Lennard parameters
lj_sigma = [2.100, 2.100, 2.365, 2.235, 2.165, 2.100, 1.6000, 1.6000, 1.6000,
            1.6000, 1.6000, 1.6000, 1.6000, 1.6000, 1.6000,  1.890, 1.890, 0.8000, 0.6000]

lj_epsilon = [-0.1200, -0.1200, -0.0486, -0.1142, -0.1811, -0.1200, -0.2384,
              -0.2384, -0.2384, -0.2384, -0.2384, -0.2384, -0.1591, -0.1591, -0.6469,
              -0.0430, -0.0430, -0.0498, -0.0498]

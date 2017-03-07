"""
A collection of common constants and forcefield parameters
"""

radians_to_degrees = 57.29577951308232  # 180/pi
degrees_to_radians = 0.017453292519943295 # pi/180


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






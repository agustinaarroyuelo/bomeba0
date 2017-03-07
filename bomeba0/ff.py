"""
Draft of a forcefield. At this point only a Lennard Jones term is implemented.
I have not made any performance test.
"""

from utils import dist
from numba import jit


def LJ(neighbors, xyz, types, par_vdw):
    """
    Lennard Jones energy term
    
    .. math::
    
    LJ_{ij} = \epsilon \left [ \left (\frac{\sigma_{ij}}{r_{ij}} \right)^{12} - 2 \left (\frac{\sigma_{ij}}{r_{ij}} \right)^{6} \right]
    
    \sigma_{ij} is the distance at which the potential reaches its minimum
    \epsilon_{ij} is the depth of the potential well
    r_{ij} is the distance between the particles
    
    Parameters
    ----------
    XXX : XXX
        XXX
    XXX : XXX
        XXX
    
    Results
    -------
    E_LJ: float
        Lennard Jones energy contribution
    
    Notes
    -----
    
    A couple of details still missing

    """

    E_vdw = 0.
    for i,j in neighbors:
        key_i = types[i]
        key_j = types[j]
        # Shold we precompute these values?
        sigma_ij = par_vdw[key_i][0] + par_vdw[key_j][0]
        epsilon_ij = (par_vdw[key_i][1] * par_vdw[key_j][1])**0.5

        r_ij = dist(xyz[i], xyz[j])

        C6 = (sigma_ij / r_ij) ** 6
		           
        E_vdw += epsilon_ij * (C6 * C6 - 2 * C6)
    return E_vdw

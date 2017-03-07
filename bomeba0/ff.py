"""
Draft of a forcefield. At this point only a Lennard Jones term is implemented.
I have not made any performance test.
"""

from utils import dist


def LJ(neighbors, xyz, types, par_vdw):
    """
    Lennard Jones energy term
    
    .. math::
    
    LJ_{ij} =  \frac{A_{ij}}{(r_{ij})^{12}} - \frac{B_{ij}}{(r_{ij})^6}
    
    A_{ij} = 4\epsilon \sigma^{12}
    B_{ij} = 4\epsilon \sigma^{6}
    
    Parameters
    ----------
    XXX : XXX
        XXX
    XXX : XXX
        XXX
    
    Results
    -------
    E_LJ: float
        Lennard jones energy contribution
    
    Notes
    -----
    
    XXX

    """

    E_vdw = 0.
    for i,j in neighbors:
        key_i = types[i]
        key_j = types[j]
        # We shold precompute these values
        # This is the Lorentz-Berthelot rule
        sigma_ij = (par_vdw[key_i][0] + par_vdw[key_j][0]) / 2
        epsilon_ij = (par_vdw[key_i][1] * par_vdw[key_j][1])**0.5

        r_ij = dist(xyz[i], xyz[j])

        # Is this the correct AMBER functional form?
        # we can replace this with any we want.
        A_ij = epsilon_ij * (sigma_ij) ** 12
        B_ij = 2 * epsilon_ij * (sigma_ij) ** 6
        #B_ij = epsilon_ij * (sigma_ij) ** 6
        
        E_vdw += (A_ij / (r_ij) ** 12) - (B_ij / (r_ij) ** 6)
    return E_vdw
 


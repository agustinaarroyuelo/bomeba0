"""
Draft of a force-field. At this point only a Lennard Jones term is implemented.
"""
from scipy.spatial import cKDTree
from numba import jit
from .utils import dist
from .constants import par_s_ij, par_eps_ij


def compute_neighbors(coords, exclusions, cut_off):
    tree_c = cKDTree(coords)
    all_pairs = tree_c.query_pairs(cut_off)
    return all_pairs - exclusions

def LJ(neighbors, xyz, elements, mode='A'):
    """
    Lennard Jones energy term

    .. math::

    LJ_{ij} = \epsilon \left [ \left (\frac{\sigma_{ij}}{r_{ij}} \right)^{12}
     - 2 \left (\frac{\sigma_{ij}}{r_{ij}} \right)^{6} \right]

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
    for i, j in neighbors:
        key_ij = elements[i] + elements[j]
        # use par_vdw without precomputing values
        #sigma_ij = par_vdw[key_i][0] + par_vdw[key_j][0]
        #epsilon_ij = (par_vdw[key_i][1] * par_vdw[key_j][1])**0.5
        # or use precomputed values
        sigma_ij = par_s_ij[key_ij]
        epsilon_ij = par_eps_ij[key_ij]

        E_vdw += _LJ(xyz, i, j, sigma_ij, epsilon_ij)
    return E_vdw
    

# convenient function just to speed up computation by 2x
@jit
def _LJ(xyz, i, j, sigma_ij, epsilon_ij):

    r_ij = dist(xyz[i], xyz[j])
    C6 = (sigma_ij / r_ij) ** 6

    return epsilon_ij * (C6 * C6 - 2 * C6)

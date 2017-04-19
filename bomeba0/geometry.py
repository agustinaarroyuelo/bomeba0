import math
import numpy as np
from .utils import mod, dot, cross, normalize
from numba import jit


def get_torsional(xyz, a, b, c, d):
    """
    Compute the torsional angle given four points
    xyz: array
        Cartesian coordinates
    a-d: int
        atom index for the four points defining the torsional
   """

    # Compute 3 vectors connecting the four points
    ba = xyz[b] - xyz[a]
    cb = xyz[c] - xyz[b]
    dc = xyz[d] - xyz[c]

    # Compute the normal vector to each plane
    u_A = cross(ba, cb)
    u_B = cross(cb, dc)

    # Measure the angle between the two normal vectors
    u_A_mod = mod(u_A)
    u_B_mod = mod(u_B)
    val = dot(u_A, u_B) / (u_A_mod * u_B_mod)
    # better fix?
    if val > 1:
        val = 1
    elif val < -1:
        val = -1
    tor_rad = np.arccos(val)

    # compute the sign
    sign = dot(u_A, dc)
    if sign > 0:
        return tor_rad
    else:
        return -tor_rad


@jit
def rotation_matrix_3d(u, theta):
    """Return the rotation matrix due to a right hand rotation of theta radians
    around an arbitrary axis/vector u.
    u: array 
        arbitrary axis/vector u
    theta: float
        rotation angle in radians
    """
    x, y, z = normalize(u)
    st = math.sin(theta)
    ct = math.cos(theta)
    mct = 1 - ct

    # filling the matrix by indexing each element is faster (with jit)
    # than writting np.array([[, , ], [, , ], [, , ]])
    R = np.zeros((3, 3))
    R[0, 0] = ct + x * x * mct
    R[0, 1] = y * x * mct - z * st
    R[0, 2] = x * z * mct + y * st
    R[1, 0] = y * x * mct + z * st
    R[1, 1] = ct + y * y * mct
    R[1, 2] = y * z * mct - x * st
    R[2, 0] = x * z * mct - y * st
    R[2, 1] = y * z * mct + x * st
    R[2, 2] = ct + z * z * mct

    return R


@jit
def set_torsional(xyz, i, j, theta_rad, idx_to_fix):
    """
    rotate a set of coordinates around the i-j axis by theta_rad
    xyz: array
        Cartesian coordinates
    i: int 
        atom i
    j: int 
        atom k
    theta_rad: float
        rotation angle in radians
    idx_to_fix: tuple
        The rotation is done in such a way that all atoms with and index larger
        than i are rotated and the rest are kept fixed. Given the way atoms are
        internally ordered the rotation could introduce distorsions on the
        geometry of the protein. This are the index of the atoms that should
        have not been rotated.
    """
    xyz_s = xyz - xyz[i]
    R = rotation_matrix_3d((xyz_s[j]), theta_rad)
    xyz[:j] = xyz_s[:j]
    xyz[j:] = xyz_s[j:] @ R
    xyz[i + idx_to_fix[0]:i + idx_to_fix[1]
        ] = xyz_s[i + idx_to_fix[0]:i + idx_to_fix[1]]
    # TODO return to original position????

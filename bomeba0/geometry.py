import math
import numpy as np
from utils import mod, dot, cross, normalize


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

    #Measure the angle between the two normal vectors
    u_A_mod = mod(u_A)
    u_B_mod = mod(u_B)
    tor_rad = np.arccos(dot(u_A, u_B) /
                             (u_A_mod * u_B_mod))
        
    # compute the sign
    sign = dot(u_A, dc)
    if sign > 0:
        return tor_rad
    else:
        return -tor_rad

       
def rotation_matrix_3d(u, theta):
    """Return the rotation matrix due to a right hand rotation of theta radians
    around an arbitrary axis/vector u.
    u: array 
        rbitrary axis/vector u
    theta: float
        rotation angle in radians
    """
    x, y, z = normalize(u)
    st = math.sin(theta)
    ct = math.cos(theta)
    mct = 1 - ct

    # given that the matrix is symmetric (except for a sing) it should be possible
    # to write it in a more efficient way
    R = np.array(
        [[  ct+x*x*mct, x*y*mct-z*st, x*z*mct+y*st],
         [y*x*mct+z*st,   ct+y*y*mct, y*z*mct-x*st],
         [z*x*mct-y*st, z*y*mct+x*st,   ct+z*z*mct]])

    return R


def set_torsional(xyz, i, j, theta):
    """
    rotate a molecule an angle theta around the i-j bond
    xyz: array
        Cartesian coordinates
    i: int 
        atom i
    j: int 
        atom j
    theta: float
        rotation angle in radians
    """
    xyz_s = xyz - xyz[i]
    R = rotation_matrix_3d((xyz_s[j]), theta)
    xyz[:i+1] = xyz_s[:i+1] 
    xyz[i+1:] = np.dot(xyz_s[i+1:], R)

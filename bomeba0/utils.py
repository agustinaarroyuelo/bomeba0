"""
A collection of common mathematical functions written for high performance with
the help of numpy and numba.
"""

import numpy as np
from numba import jit


@jit
def dist(p, q):
    """
    Compute distance between two 3D vectors
    p: array
        Cartesian coordinates for one of the vectors
    q: array
        Cartesian coordinates for one of the vectors
    """
    return ((p[0] - q[0])**2 + (p[1] - q[1])**2 + (p[2] - q[2])**2)**0.5


@jit
def dot(p, q):
    """
    Compute dot product between two 3D vectors
    p: array
        Cartesian coordinates for one of the vectors
    q: array
        Cartesian coordinates for one of the vectors
    """
    return p[0] * q[0] + p[1] * q[1] + p[2] * q[2]


@jit
def cross(p, q):
    """
    Compute cross product between two 3D vectors
    p: array
        Cartesian coordinates for one of the vectors
    q: array
        Cartesian coordinates for one of the vectors
    """
    xyz = np.zeros(3)
    xyz[0] = p[1] * q[2] - p[2] * q[1]
    xyz[1] = p[2] * q[0] - p[0] * q[2]
    xyz[2] = p[0] * q[1] - p[1] * q[0]
    return xyz


@jit
def mod(p):
    """
    Compute modulus of 3D vector
    p: array
        Cartesian coordinates
    """
    return (p[0]**2 + p[1]**2 + p[2]**2)**0.5


@jit
def normalize(p):
    """
    Compute a normalized 3D vector
    p: array
        Cartesian coordinates
    """
    return p / mod(p)


@jit
def perp_vector(p, q, r):
    """
    Compute perpendicular vector to (p-q) and (r-q) centered in q.
    """
    v = cross(q - r, q - p)
    return v / mod(v) + q


def get_angle(a, b, c):
    """
    Compute the angle given 3 points
    xyz: array
        Cartesian coordinates
    a-c: int
        atom index for the three points defining the angle
    """

    ba = a - b
    cb = c - b

    ba_mod = mod(ba)
    cb_mod = mod(cb)
    val = dot(ba, cb) / (ba_mod * cb_mod)
    # better fix?
    if val > 1:
        val = 1
    elif val < -1:
        val = -1

    return np.arccos(val)

# this function also exist inside geometry module
# The only diferene is that one neex and array xyz and the other do not
def get_torsional(a, b, c, d):
    """
    Compute the torsional angle given four points
    a-d: int
        atom index for the four points defining the torsional
   """
    
    # Compute 3 vectors connecting the four points
    ba = b - a
    cb = c - b
    dc = d - c
    
    # Compute the normal vector to each plane
    u_A = cross(ba, cb)
    u_B = cross(cb, dc)

    #Measure the angle between the two normal vectors
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

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
    return ((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2)**0.5

@jit
def dot(p, q):
    """
    Compute dot product between two 3D vectors
    p: array
        Cartesian coordinates for one of the vectors
    q: array
        Cartesian coordinates for one of the vectors
    """
    return p[0]*q[0] + p[1]*q[1] + p[2]*q[2]

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
def mod(x):
    """
    Compute modulus of a 3D vector
    x: array
        Cartesian coordinates for a vector
    """
    return (x[0]**2 + x[1]**2 + x[2]**2)**0.5


@jit
def m_ang(u_A, u_B):
    """
    FIXME I am not using this function!!!
    """
    return np.arccos(dot(u_A, u_B) / (mod(u_A) * mod(u_B)))

@jit
def normalize(p):
    """
    Compute a normalized 3D vector
    p: array
        Cartesian coordinates
    """
    return p/(dot(p, p))**0.5

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

# We are not using this function!
@jit
def perp_vector(p, q, r):
    """
    Compute perpendicular vector to (p-q) and (r-q) centered in q.
    """
    return norm(cross(q - r, q - p)) + q

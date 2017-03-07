from numpy.testing import assert_almost_equal
import numpy as np
from ..utils import dist, dot, cross, mod

p = np.array([3.115, 1.287, -0.589])
q = np.array([6.072, 2.231, 1.554])


def test_dist():
    distance = np.linalg.norm(p - q)
    assert_almost_equal(dist(p, q), distance)
   
def test_dot():
    assert_almost_equal(dot(p, q), np.dot(p, q))
    
def test_cross():
    assert_almost_equal(cross(p, q), np.cross(p, q))
    
def test_mod():
    assert_almost_equal(mod(p), np.linalg.norm(p))

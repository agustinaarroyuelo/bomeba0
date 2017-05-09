import numpy as np
from numpy.testing import assert_almost_equal
from ..biomolecules import Protein
from ..OBB import _little_boxes

seq_reference = 'AAA'
prot = Protein(seq_reference)


def test_little_boxes():
    l = np.vstack((prot.at_coords(0, 'N'), prot.at_coords(0, 'H')))
    m = np.vstack((prot.at_coords(0, 'sc'), prot.at_coords(0, 'CA'), prot.at_coords(0, 'HA')))
    n = np.vstack((prot.at_coords(0, 'C'), prot.at_coords(0, 'O'), prot.at_coords(1, 'N'), prot.at_coords(1, 'H')))
    o = np.vstack((prot.at_coords(1, 'sc'), prot.at_coords(1, 'CA'), prot.at_coords(1, 'HA')))
    p = np.vstack((prot.at_coords(1, 'C'), prot.at_coords(1, 'O'), prot.at_coords(2, 'N'), prot.at_coords(2, 'H')))
    q = np.vstack((prot.at_coords(2, 'sc'), prot.at_coords(2, 'CA'), prot.at_coords(2, 'HA')))
    r = np.vstack((prot.at_coords(2, 'C'), prot.at_coords(2, 'O')))
    sel_0 = [l, m, n, o, p, q, r]
    boxes = _little_boxes(prot)
    xyz = prot.coords
    sel_1 = [xyz.take(i, 0) for i in  boxes]
    for i,j in zip(sel_0, sel_1):
        assert_almost_equal(i, j)

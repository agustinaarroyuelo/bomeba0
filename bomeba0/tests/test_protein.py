import numpy as np
from numpy.testing import assert_almost_equal
from ..biomolecules import Protein

seq_reference = 'GG'
prot = Protein(seq_reference)


def test_coords():
    coords = prot.coords
    # this test is valid for GG
    ref_coords = np.array([[-3.20397075,  1.31512566, -0.18621926],
       [-1.77897075,  1.43212566, -0.48221926],
       [-0.94997075,  0.72412566,  0.56178074],
       [-1.17692165,  0.84364961,  1.76862676],
       [-3.56697075,  0.78112566,  0.67978074],
       [-1.52697075,  2.45112566, -0.49421926],
       [-1.57497075,  0.95512566, -1.45921926],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.8019035 , -0.68485848,  1.00987606],
       [ 2.27505457, -0.56390502,  0.70387415],
       [ 2.68965127,  0.04644274, -0.28499146],
       [ 0.1984067 , -0.1049676 , -1.05664449],
       [ 0.52988113, -1.69875279,  1.01545751],
       [ 0.61920658, -0.21329074,  1.99370516]])
    assert_almost_equal(coords, ref_coords)


def test_exclusions():
    # the first 13 tuples correspond to 1-2 interactions (bonds),
    # the rest are 1-3 interactions.
    # this test is valid for GG
    reference = {(0, 1), (0, 4), (1, 5), (1, 6), (1, 2), (2, 3), (2, 7),
                 (7, 11), (7, 8), (8, 9), (8, 12), (8, 13), (9, 10), (1, 4),
                 (0, 2), (0, 5), (0, 6), (2, 5), (2, 6), (1, 3), (1, 7),
                 (3, 7), (2, 11), (2, 8), (5, 6), (7, 12), (7, 13), (7, 9),
                 (8, 11), (9, 13), (9, 12), (8, 10), (12, 13)}
    exclusions = prot._exclusions
    assert reference == exclusions


def test_names():
    # this test is valid for GG
    names = prot._names
    assert ['N', 'CA', 'C', 'O', 'H', 'HA2', 'HA3'] * 2 == names


def test_offsets():
    # this test is valid for GG
    offsets = prot._offsets
    assert [0, 7, 14, 21] == offsets
    
def test_set_get_torsionals():
    # create a molecule set the backbone torsional angles and then check that 
    # the actual values corresponds to the set ones.
    poly_gly = Protein('GGGGGG')

    for i in range(len(poly_gly)):
        poly_gly.set_phi(i, -60.)
        poly_gly.set_psi(i, -40.)

    for i in range(1, len(poly_gly)-1):
        assert_almost_equal(poly_gly.get_phi(i), -60., 5)
        assert_almost_equal(poly_gly.get_psi(i), -40., 5)
        
    assert prot.get_phi(0) is np.nan
    assert prot.get_psi(len(poly_gly)-1) is np.nan
    
def test_at_coords():
    assert_almost_equal(prot.at_coords(1), prot.coords[7:])
    assert_almost_equal(prot.at_coords(1, 'N'), prot.coords[7])
    assert_almost_equal(prot.at_coords(1, 'bb'), prot.coords[7:])
    assert len(prot.at_coords(1, 'sc')) == 0 

import numpy as np
from numpy.testing import assert_almost_equal
import filecmp
from ..biomolecules import Protein

seq_reference = 'GG'
prot = Protein(seq_reference)


def test_coords():
    coords = prot.coords
    # this test is valid for GG
    ref_coords = np.array([[-3.3000322 ,  1.36752836, -0.25581245],
       [-1.8470322 ,  1.45852836, -0.51581245],
       [-1.0000322 ,  0.76252836,  0.52818755],
       [-1.18048954,  0.85541089,  1.72376188],
       [-3.4680322 ,  0.84252836,  0.60218755],
       [-1.5350322 ,  2.50552836, -0.53481245],
       [-1.6150322 ,  1.00352836, -1.47581245],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.82743155, -0.67992014,  1.01988021],
       [ 2.31479446, -0.58676803,  0.7537313 ],
       [ 2.8018398 , -0.00933498, -0.19492454],
       [ 0.3356148 , -0.2350187 , -0.93388539],
       [ 0.57534648, -1.74239314,  1.05884493],
       [ 0.64572246, -0.23790773,  1.99663803]])
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
    
def test_protein():
    prot.dump_pdb('test_1')
    prot2 = Protein(pdb='test_1.pdb')
    prot2.dump_pdb('test_2')
    assert filecmp.cmp('test_1.pdb', 'test_2.pdb')
    

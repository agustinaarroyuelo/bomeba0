import numpy as np
from numpy.testing import assert_almost_equal
from ..biomolecules import Protein

seq_reference = 'GG'
prot = Protein(seq_reference)


def test_coords():
    coords = prot.coords
    # this test is valid for GG
    ref_coords = np.array([[-1.19500005,  0.20100001, -0.206],
                           [0.23,  0.31799999, -0.50199997],
                           [1.05900002, -0.38999999,  0.542],
                           [0.54500002, -0.97500002,  1.49899995],
                           [-1.55799997, -0.333,  0.66000003],
                           [0.48199999,  1.33700001, -0.514],
                           [0.43399999, -0.15899999, -1.47899997],
                           [2.34725648, -0.28422738,  0.27440431],
                           [3.14916003, -0.96908586,  1.28428038],
                           [4.6223111, -0.84813241,  0.97827853],
                           [5.03690778, -0.23778461, -0.01058703],
                           [2.79722815,  0.22759492, -0.56371527],
                           [2.87713764, -1.98298019,  1.28986183],
                           [2.96646307, -0.49751815,  2.26810948]])
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
    assert ['N', 'CA', 'C', 'O', 'H', '1HA', '2HA'] * 2 == names


def test_offsets():
    # this test is valid for GG
    offsets = prot._offsets
    assert [0, 7, 14, 21] == offsets

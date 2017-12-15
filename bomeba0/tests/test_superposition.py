from numpy.testing import assert_almost_equal
from ..superposition.super import rmsd_fit
from ..molecules.protein import Protein

prot0 = Protein('GAG')
prot1 = Protein('GAG')
prot2 = Protein('GAG', ss='helix')

# very basic test, we should improve it.
def test_rmsd_fit():
    rmsd = rmsd_fit(prot0, prot1)
    assert_almost_equal(rmsd, 0)
    rmsd = rmsd_fit(prot0, prot2)
    assert_almost_equal(rmsd, 1.95, decimal=2)

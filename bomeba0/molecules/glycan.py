"""
Glycan class
still very buggy, idiosyncratic, fragile and completely ad-hoc
"""
import numpy as np
import pandas as pd
from .biomolecules import Biomolecule
from ..constants import constants
from ..templates.glycans import templates_gl
from ..pdbIO import _builder_from_pdb_gl
from ..utils import get_torsional
from ..geometry import set_torsional

class Glycan(Biomolecule):
    """Glycan object"""

    def __init__(self, pdb=None, linkages=None):
        """initialize a new Glycan from a PDB

        Parameters
        ----------
        pdf : file
            Protein data bank file.
            For the moment this will only work with "nice" files:        
        Returns
        ----------
        Glycan object
        """
        if pdb is not None:
            (self.sequence,
             self.coords,
             self._names,
             self._elements,
             self.occupancies,
             self.bfactors,
             self._offsets,
             self._exclusions) = _builder_from_pdb_gl(pdb, 'glycan',
                                                      regularize=True,                                                        
                                                      linkages=linkages)

            self._rotation_indices = _get_rotation_indices_gl(self,
                                                              linkages=linkages)
        else:
            "Please provide a sequence or a pdb file"

    def at_coords(self, resnum, selection=None):
        """
        Returns the coordinate of an specified residue and atom (optionally)

        Parameters
        ----------
        resnum : int
            residue number from which to obtain the coordinates
        selection : string or None
            selection from which to obtain the coordinates. If none is provided
            it will return the coordinates of the whole residue (default). Use
            a valid atom name.

        Returns
        ----------
        coords: array
            Cartesian coordinates of a given residue or a subset of atoms in a
            given residue.
        """
        offsets = self._offsets
        offset_0, offset_1 = offsets[resnum], offsets[resnum + 1]
        rescoords = self.coords[offset_0: offset_1]

        if selection is None:
            return rescoords
        else:
            resname = self.sequence[resnum]
            resinfo = templates_gl[resname]
            idx = resinfo.atom_names.index(selection)
            return rescoords[idx]

    def get_phi(self, resnum):
        """
        Compute the dihedral angle phi (OR-C1-O'x-C'x)

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        """
        if resnum < len(self) - 1:
            coords = self.coords
            _, _, _, m, n, o, p = self._rotation_indices[resnum]['phi']
            a = coords[m]
            b = coords[n]
            c = coords[o]
            d = coords[p]
            return get_torsional(a, b, c, d) * constants.radians_to_degrees
        else:
            return np.nan

    def get_psi(self, resnum):
        """
        Compute the dihedral angle psi (C1-O'x-C'x-C'x-1)

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        """
        if resnum < len(self) - 1:
            coords = self.coords
            _, _, _, m, n, o, p = self._rotation_indices[resnum]['psi']
            a = coords[m]
            b = coords[n]
            c = coords[o]
            d = coords[p]
            return get_torsional(a, b, c, d) * constants.radians_to_degrees
        else:
            return np.nan

    def set_phi(self, resnum, theta):
        """
        set the phi torsional angle (OR-C1-O'x-C'x) to the value theta

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        theta : float
            value of the angle to set in degrees
        """
        if resnum + 1 < len(self):
            theta_rad = (self.get_phi(resnum) - theta) * \
                constants.degrees_to_radians
            xyz = self.coords
            i, j, idx_rot, _, _, _, _ = self._rotation_indices[resnum]['phi']
            set_torsional(xyz, i, j, idx_rot, theta_rad)

    def set_psi(self, resnum, theta):
        """
        set the psi torsional angle (C1-O'x-C'x-C'x-1) to the value theta

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        theta : float
            value of the angle to set in degrees
        """
        if resnum + 1 < len(self):
            theta_rad = (self.get_psi(resnum) - theta) * \
                constants.degrees_to_radians
            xyz = self.coords
            i, j, idx_rot, _, _, _, _ = self._rotation_indices[resnum]['psi']
            set_torsional(xyz, i, j, idx_rot, theta_rad)


    def get_torsionals(self,  n_digits=2):
        """
        Compute all phi, psi and chi torsional angles of a given molecule

        Parameters
        ----------
        n_digits : int
            Number of decimal digits used to round the torsional values
            (default 2 digits).

        Returns
        ----------
        DataFrame with the glycan sequence and torsional angles.

        """
        all_tors = []
        for i, res in enumerate(self.sequence[:-1]):
            tors = []
            tors.append(round(self.get_phi(i), n_digits))
            tors.append(round(self.get_psi(i), n_digits))
            all_tors.append([res] + tors)
        else:
            labels = ['res', 'phi', 'psi']
        df = pd.DataFrame.from_records(all_tors, columns=labels)
        return df
        
def _get_rotation_indices_gl(self, linkages):
    """
    Precompute indices that are then used to rotate along the i-j axis only the
    proper portion of the coordinates array. Works only for glycans.
    
    
    Returns
    rotation_indices : dict of dict of tuples 
        dictionary of residue numbers, dicctionary of torsionals (phi or psi), 
    7-element tuple: The first 3 elements are atom i (int), atom j (int)
    indices of the atoms that will be rotated (array). Last four elements atom
    ids for the atoms defining a torsional angle.
    """
    rotation_indices = []
    lenght = len(self.coords)
    offsets = self._offsets
    seq = self.sequence
    rotation_indices = {}
    for resnum in range(0, len(self) - 1):
        res = resnum
        d = {}
        linkage = linkages[resnum]
        if isinstance(linkage, tuple):
            resnum = linkage[0]
            linkage = linkage[1]     
        if linkage > 0:  # forward reading
            this = offsets[resnum]  # index of C1
            post = offsets[resnum + 1]
            resname_this = seq[resnum]
            resname_post = seq[resnum + 1]
            pre_idx_rot = list(range(post, lenght))
        else:  # backward reading
            this = offsets[resnum + 1]  # index of C1
            post = offsets[resnum]
            resname_this = seq[resnum + 1]
            resname_post = seq[resnum]
            pre_idx_rot = list(range(0, this))
            linkage = abs(linkage)

        template_at_names_this = templates_gl[resname_this].atom_names
        template_at_names_post = templates_gl[resname_post].atom_names
        OR_idx = template_at_names_this.index('OR')
        O_idx = template_at_names_post.index('O{}'.format(linkage))
        C_idx = template_at_names_post.index('C{}'.format(linkage))
        # following IUPAC for 1-1 bonds use C'x+1 instead of C'x-1
        # check http://www.glycosciences.de/spec/ppc/ and
        # http://www.chem.qmul.ac.uk/iupac/2carb/ for details
        if linkage == 1:
            fourth_point = linkage + 1
        else:
            fourth_point = linkage - 1
        C__idx = template_at_names_post.index('C{}'.format(fourth_point))

        ###  phi  ###
        j = post + O_idx
        l = post + C_idx
        # making idx_rot an array makes rotation faster later
        idx_rot = np.asarray(pre_idx_rot)
        # the terms of the tuple are the indices of:
        # (two atoms defining the axis of rotation, the atoms that will be rotated)
        # and (OR-C1-O'x-C'x)
        d['phi'] = this, j, idx_rot, this + OR_idx, this, j, l


        ### psi ###
        pre_idx_rot.remove(j)
        #if linkages[resnum] > 0:
        #    pre_idx_rot.remove(j)
        #else:
        #    pre_idx_rot.append(j)
        # making idx_rot an array makes rotation faster later
        idx_rot = np.asarray(pre_idx_rot)
        # the terms of the tuple are the indices of:
        # (two atoms defining the axis of rotation, the atoms that will be rotated)
        # (C1-O'x-C'x-C'x-1)
        d['psi'] = j, l, idx_rot, this, j, l, post + C__idx
        rotation_indices[res] = d
    return rotation_indices

"""
Protein class
"""
import numpy as np
import pandas as pd
from .biomolecules import Biomolecule
from ..templates.aminoacids import templates_aa, three_to_one_aa
from ..constants import constants, rotamers
from ..pdbIO import _prot_builder_from_seq, _pdb_parser
from ..utils import get_torsional
from ..geometry import set_torsional

class Protein(Biomolecule):
    """Protein object"""

    def __init__(self, sequence=None, pdb=None, ss='strand', torsionals=None,
                 regularize=False):
        """initialize a new protein from a sequence of amino acids

        Parameters
        ----------
        sequence : str
            protein sequence using one letter code. Accepts lower and uppercase
            sequences
            example 'GAD'
        pdb : file
            Protein data bank file.
            For the moment this will only work with "nice" files. Like:
            * files generated with bomeba
            * Files with only one chain
            * Files without missing residues/atoms
            * Files with only one model
        ss : str
            secondary structure to initialize the protein. Two options allowed
            'strand' (-135, 135)
            'helix' (-60, -40)
            This argument is only valid when a sequence is passed and not when
            the structure is generated from a pdb file
        torsionals : list of tuples
            list of phi and psi torsionals angles one tuple per residue in
            sequence. Works together with sequence.
        regularize : bool
            Whether to regularize the structure. Regularization is needed for
            some computations like setting torsionals angles. Only works with
            the `pdb` argument. If a sequence is passed it will be always
            regularized, since the protein will be constructed from the
            templates.
        Returns
        ----------
        Protein object
        """
        if sequence is not None:
            self.sequence = sequence.upper()
            (self.coords,
             self._names,
             self._elements,
             self.occupancies,
             self.bfactors,
             self._offsets,
             self._exclusions) = _prot_builder_from_seq(self.sequence)

            self._rotation_indices = _get_rotation_indices_prot(self)

            if torsionals is not None:
                for idx, val in enumerate(torsionals):
                    i = float(val[0])
                    j = float(val[1])
                    self.set_phi(idx, i)
                    self.set_psi(idx, j)
            else:
                if ss == 'strand':
                    for i in range(len(self)):
                        self.set_phi(i, -135)
                        self.set_psi(i, 135)
                elif ss == 'helix':
                    for i in range(len(self)):
                        self.set_phi(i, -60)
                        self.set_psi(i, -40)

        elif pdb is not None:
            (self._names,
             self.sequence,
             self.coords,
             self.occupancies,
             self.bfactors,
             self._elements,
             self._offsets) = _pdb_parser(pdb, three_to_one_aa)
 
            self._exclusions = []
            self._rotation_indices = _get_rotation_indices_prot(self)

            if regularize:
                torsionals = self.get_torsionals(n_digits=4)
                tors = torsionals[['phi', 'psi', 'omega']].values

                (self.coords,
                 self._names,
                 self._elements,
                 self.occupancies,
                 self.bfactors,
                 self._offsets,
                 self._exclusions) = _prot_builder_from_seq(self.sequence)
                self._rotation_indices = _get_rotation_indices_prot(self)

                for idx, val in enumerate(tors):
                    self.set_phi(idx, val[0])
                    self.set_psi(idx, val[1])
                    self.set_omega(idx, val[2])

        else:
            "Please provide a sequence or a pdb file"

    def __repr__(self):
        """
        ToDo do something useful
        """
        return 'I am a protein'

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
            'sc' for the sidechain, 'bb' for the backbone or a valid atom name.

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
            # this only works for structures from sequence
            resinfo = templates_aa[resname]
            if selection == 'sc':
                idx = resinfo.sc
            elif selection == 'bb':
                idx = resinfo.bb
            else:
                atom_names = self._names[offset_0: offset_1]
                idx = atom_names.index(selection)
            return rescoords[idx]


    def get_phi(self, resnum):
        """
        Compute phi torsional angle

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        """
        # C(i-1),N(i),Ca(i),C(i)
        if resnum != 0:
            coords = self.coords
            prev = self._offsets[resnum - 1]
            this = self._offsets[resnum]

            a = coords[prev + 2]
            b = coords[this]
            c = coords[this + 1]
            d = coords[this + 2]
            return get_torsional(a, b, c, d) * constants.radians_to_degrees
        else:
            return np.nan

    def get_psi(self, resnum):
        """
        Compute psi torsional angle

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        """
        # N(i),Ca(i),C(i),N(i+1)
        if resnum < len(self) - 1:
            coords = self.coords
            post = self._offsets[resnum + 1]
            this = self._offsets[resnum]
            a = coords[this]
            b = coords[this + 1]
            c = coords[this + 2]
            d = coords[post]
            return get_torsional(a, b, c, d) * constants.radians_to_degrees
        else:
            return np.nan

    def get_omega(self, resnum):
        """
        Compute omega torsional angle

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        """
        # Ca(i),C(i),N(i+1), Ca(i+1)
        if resnum < len(self) - 1:
            coords = self.coords
            post = self._offsets[resnum + 1]
            this = self._offsets[resnum]
            a = coords[this + 1]
            b = coords[this + 2]
            c = coords[post]
            d = coords[post + 1]
            return get_torsional(a, b, c, d) * constants.radians_to_degrees
        else:
            return np.nan

    def get_chi(self, resnum, chi_num):
        """
        Compute chi torsional angle

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        chi_num : int
            number of chi, some residues have more than one chi torsional:
        """
        coords = self.coords
        seq = self.sequence
        resname = seq[resnum]
        if resname not in rotamers[chi_num]:
            return np.nan
        else:
            this = self._offsets[resnum]
            if chi_num == 1:
                a = coords[this]
                b = coords[this + 1]
                c = coords[this + 4]
                d = coords[this + 5]
            elif chi_num == 2:
                a = coords[this + 1]
                b = coords[this + 4]
                c = coords[this + 5]
                d = coords[this + 6]
            elif chi_num >= 3:
                a = coords[this + chi_num + 1]
                b = coords[this + chi_num + 2]
                c = coords[this + chi_num + 3]
                d = coords[this + chi_num + 4]
            return get_torsional(a, b, c, d) * constants.radians_to_degrees

    def get_torsionals(self, sidechain=True, n_digits=2):
        """
        Compute all phi, psi and chi torsional angles of a given molecule

        Parameters
        ----------
        sidechain : Boolean
            whether to compute all torsional angles including the chi angles or
            only backbone ones. The default (True) is to include the sidechain.
        n_digits : int
            Number of decimal digits used to round the torsional values
            (default 2 digits).

        Returns
        ----------
        DataFrame with the protein sequence and torsional angles.

        """
        all_tors = []
        for i, aa in enumerate(self.sequence):
            tors = []
            tors.append(round(self.get_phi(i), n_digits))
            tors.append(round(self.get_psi(i), n_digits))
            tors.append(round(self.get_omega(i), n_digits))
            if sidechain:
                for j in range(1, 6):
                    tors.append(round(self.get_chi(i, j), n_digits))
            all_tors.append([aa] + tors)
        if sidechain:
            labels = ['aa', 'phi', 'psi', 'omega', 'chi1',
                      'chi2', 'chi3', 'chi4', 'chi5']
        else:
            labels = ['aa', 'phi', 'psi', 'omega']
        df = pd.DataFrame.from_records(all_tors, columns=labels)
        return df

    def set_phi(self, resnum, theta):
        """
        set the phi torsional angle C(i-1),N(i),Ca(i),C(i) of residue `resnum`
        to the value `theta`.

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        theta : float
            value of the angle to set in degrees
        """
        if resnum != 0:
            theta_rad = (self.get_phi(resnum) - theta) * \
                constants.degrees_to_radians
            xyz = self.coords
            i, j, idx_rot = self._rotation_indices[resnum]['phi']
            set_torsional(xyz, i, j, idx_rot, theta_rad)

    def set_psi(self, resnum, theta):
        """
        set the psi torsional angle N(i),Ca(i),C(i),N(i+1) of residue `resnum`
        to the value `theta`.

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
            i, j, idx_rot = self._rotation_indices[resnum]['psi']
            set_torsional(xyz, i, j, idx_rot, theta_rad)

    def set_omega(self, resnum, theta):
        """
        Set the omega torsional angle Ca(i),C(i),N(i+1),Ca(i+1) of residue
        `resnum` to the value `theta`.

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        theta : float
            value of the angle to set in degrees
        """
        if resnum + 1 < len(self):
            theta_rad = (self.get_omega(resnum) - theta) * \
                constants.degrees_to_radians
            xyz = self.coords
            i, j, idx_rot = self._rotation_indices[resnum]['omega']
            set_torsional(xyz, i, j, idx_rot, theta_rad)
            
def _get_rotation_indices_prot(self):
    """
    Precompute indices that are then used to rotate only the proper portion of
    the coordinates array. Works only for proteins.
    """
    rotation_indices = []
    lenght = len(self.coords)
    for resnum in range(0, len(self)):
        d = {}
        ###      phi        ###
                 ###    psi      ###
                      ###    omega      ###
        # C(h),N(i),Ca(i),C(i),N(j) Ca(j)
        N_i = self._offsets[resnum]
        Ca_i = N_i + 1
        C_i = N_i + 2
        N_j = self._offsets[resnum + 1]
        resname = self.sequence[resnum]
        a = list(range(Ca_i, lenght))
        try:
            H = templates_aa[resname].atom_names.index('H')
            a.remove(N_i + H)  # H atom should not be rotated
        except ValueError:
            pass
        idx_rot = np.array(a)  # rotation are faster if idx_rot is an array
        d['phi'] = N_i, Ca_i, idx_rot
        a = list(range(N_j, lenght))
        idx_rot_omega = np.array(a)
        C = N_i + templates_aa[resname].atom_names.index('C')
        O = N_i + templates_aa[resname].atom_names.index('O')
        a.extend((C, O))  # The C and O atoms from this residue should rotate
        idx_rot = np.array(a)  # rotation are faster if idx_rot is an array
        d['psi'] = Ca_i, C_i, idx_rot
        d['omega'] = C_i, N_j, idx_rot_omega
        ###  chi  ###
        rotation_indices.append(d)
    return rotation_indices

import numpy as np
import pandas as pd
from .residues import templates_aa, one_to_three_aa, three_to_one_aa
from .glycans import templates_gl, one_to_three_gl, three_to_one_gl
from .utils import mod, perp_vector, get_angle, get_torsional
from .geometry import rotation_matrix_3d, set_torsional, set_torsional_tmp
from .constants import constants
from .ff import compute_neighbors, LJ


class TestTube():
    """
    this is a "container" class instantiated only once (Singleton)
    """
    _instance = None
    def __new__(cls, solvent=None, temperature=298, force_field='simple_lj',
    *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TestTube, cls).__new__(
                                cls, *args, **kwargs)
            cls.solvent = solvent
            cls.temperature = temperature
            cls.force_field = force_field
            cls.molecules = []
        return cls._instance

    def energy(self):
        """
        Compute the energy of the system.
        ToDo: It should be possible to compute partial energy,
        like without solvent or excluding molecules.
        At the moment this method lives in Protein class
        """
        pass

    def add(self, name):
        """
        add molecules to TestTube
        """
        molecules = self.molecules
        if name in molecules:
            print('We already have a copy of {:s} in the test tube!'.format(name))
        else:
            molecules.append(name)
    
    def remove(self, name):
        """
        remove molecules from TestTube
        """
        molecules = self.molecules
        if name in molecules:
            molecules.remove(name)


class Biomolecule():
    """Base class for biomolecules"""
    def __init__(self):
        self.sequence
        self.coords 
        self._names
        self._elements
        self._offsets
        self._exclusions

    def __len__(self):
        return len(self.sequence)

    def get_torsionals(self):
        raise NotImplementedError()


    def dump_pdb(self, filename, b_factor=None):
        """
        Write a molecule object to a pdb file

        Parameters
        ----------
        filename : string
            name of the file without the pdb extension
        b_factor : list optional
            list of values to fill the b-factor column. one value per atom
        """
        if isinstance(self, Protein):
            one_to_three = one_to_three_aa
            templates = templates_aa
        elif isinstance(self, Glycan):
            one_to_three = one_to_three_gl
            templates = templates_gl
        
        coords = self.coords
        names = self._names
        elements = self._elements
        sequence = self.sequence
        
        if b_factor is None:
            b_factor = [0.] * len(coords)

        rep_seq_nam = []
        rep_seq = []
        for idx, aa in enumerate(sequence):
            lenght = templates[aa].offset
            seq_nam = aa * lenght
            res = [str(idx + 1)] * lenght
            rep_seq_nam.extend(seq_nam)
            rep_seq.extend(res)

        fd = open('{}.pdb'.format(filename), 'w')
        for i in range(len(coords)):
            serial = str(i + 1)
            name = names[i]
            if len(name) < 4:
                name = ' ' + name
            resname = one_to_three[rep_seq_nam[i]]
            resseq = rep_seq[i]
            line = "ATOM {:>6s} {:<4s} {:>3s} A{:>4s}    {:8.3f}{:8.3f}{:8.3f}  1.00 {:5.2f}           {:2s} \n".format(serial, name, resname, resseq, *coords[i], b_factor[i], elements[i])
            fd.write(line)
        fd.close()


    def energy(self, cut_off=6., neighbors=None):
        """
        Compute the internal energy of a molecule using a pair-wise 
        Lennard-Jones potential.

        Parameters
        ----------
        cut_off : float
            Only pair of atoms closer than cut_off will be used to compute the
            energy. Default 6. Only valid when neighbors is None.
        neighbors: set of tuples
            Pairs of atoms used to compute the energy. If None (default) the
            list of neighbors will be computed using a KD-tree (from scipy),
            see ff.compute_neighbors for details.

        Returns
        ----------
        energy : float:
            molecular energy in Kcal/mol

        """
        coords = self.coords
        if neighbors is None:
            neighbors = compute_neighbors(coords, self._exclusions, cut_off)
        energy = LJ(neighbors, coords, self._elements)
        return energy


class Protein(Biomolecule):
    """Protein object"""
    def __init__(self, sequence=None, pdb=None, ss='strand', tor_list=None):
        """initialize a new protein from a sequence of amino acids

        Parameters
        ----------
        sequence : str
            protein sequence using one letter code. Accepts lower and uppercase
            sequences
            example 'GAD'
        pdf : file
            Protein data bank file.
            For the moment this will only work with "nice" files. Like:
            * files generated with bomeba
            * x-ray files from the PDB
            * Files without missing residues/atoms
            For NMR files is not able to recognize the different models.
        ss : str
            secondary structure to initialize the protein. Two options allowed
            'strand' (-135, 135)
            'helix' (-60, -40)
            This argument is only valid when a sequence is passed and not when
            the structure is generated from a pdb file
        
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

            if tor_list is not None:
                for idx, val in enumerate(tor_list):
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
            (self.sequence,
            self.coords,
            self._names,
            self._elements,
            self.occupancies,
            self.bfactors,
            self._offsets,
            self._exclusions) = _builder_from_pdb(pdb, 'protein')
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
            'sc' for the sidechain, 'bb' for the backbone or a valid atom name.

        Returns
        ----------
        coords: array
            Cartesian coordinates of a given residue or a subset of atoms in a
            given residue.
        """
        offsets = self._offsets
        offset_0, offset_1 = offsets[resnum], offsets[resnum + 1]
        rescoords = self.coords[offset_0 : offset_1]
        
        if selection is None:
            return rescoords
        else:
            resname = self.sequence[resnum]
            resinfo = templates_aa[resname]
            if selection == 'sc':
                idx = resinfo.sc
            elif selection == 'bb':    
                idx = resinfo.bb
            else:
                idx = resinfo.atom_names.index(selection)
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
        if resnum + 1 < len(self):
            coords = self.coords
            next = self._offsets[resnum + 1]
            this = self._offsets[resnum]
            a = coords[this]
            b = coords[this + 1]
            c = coords[this + 2]
            d = coords[next]
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
        return np.nan


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
            if sidechain:
                for j in range(5):
                    tors.append(round(self.get_chi(i, j), n_digits))
            all_tors.append([aa] + tors)
        if sidechain:
            labels = ['aa', 'phi', 'psi', 'chi1', 'chi2', 'chi3', 'chi4', 'chi5']
        else:
            labels = ['aa', 'phi', 'psi']
        df = pd.DataFrame.from_records(all_tors, columns=labels)
        return df


    def set_phi(self, resnum, theta):
        """
        set the phi torsional angle C(i-1),N(i),Ca(i),C(i) to the value theta

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        theta : float
            value of the angle to set in degrees
        """
        if resnum != 0:
            theta_rad = (self.get_phi(resnum) - theta) * constants.degrees_to_radians
            xyz = self.coords
            i, j, idx_rot = self._rotation_indices[resnum]['phi']
            set_torsional_tmp(xyz, i, j, idx_rot, theta_rad)
            

    def set_psi(self, resnum, theta):
        """
        set the psi torsional angle N(i),Ca(i),C(i),N(i+1) to the value theta

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        theta : float
            value of the angle to set in degrees
        """
        if resnum + 1 < len(self):
            theta_rad = (self.get_psi(resnum) - theta) * constants.degrees_to_radians
            xyz = self.coords
            #i = self._offsets[resnum] + 1
            #j = i + 1
            # We have made a rotation starting from the next residue and we
            # left C and O atoms unrotated, now we fix this
            #resname = self.sequence[resnum]
            #idx_to_fix = (3, templates_aa[resname].offset - 1)
            #set_torsional(xyz, i, j, theta_rad, idx_to_fix)
            i, j, idx_rot = self._rotation_indices[resnum]['psi']
            set_torsional_tmp(xyz, i, j, idx_rot, theta_rad)


class Glycan(Biomolecule):
    """Glycan object"""
    def __init__(self, pdb=None):
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
            self._exclusions) = _builder_from_pdb(pdb, 'glycan')
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
        rescoords = self.coords[offset_0 : offset_1]
        
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
        if resnum + 1 < len(self):
            coords = self.coords
            this = self._offsets[resnum]
            next = self._offsets[resnum + 1]
            resname = self.sequence[resnum]
            O_idx = templates_gl[resname].atom_names.index('OR')

            a = coords[this + O_idx]
            b = coords[this]
            c = coords[next + 4] # true only for bond 1-3
            d = coords[next + 3] # true only for bond 1-3
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
        if resnum + 1 < len(self):
            coords = self.coords
            this = self._offsets[resnum]
            next = self._offsets[resnum + 1]

            a = coords[this]
            b = coords[next + 4] # true only for bond 1-3
            c = coords[next + 3] # true only for bond 1-3
            d = coords[next + 1] # true only for bond 1-3
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
            theta_rad = (self.get_phi(resnum) - theta) * constants.degrees_to_radians
            xyz = self.coords
            i = self._offsets[resnum]  # index of C1
            resname = self.sequence[resnum + 1]
            O_idx = templates_gl[resname].atom_names.index('O3')
            k = self._offsets[resnum + 1]
            j = k + O_idx   # index of O'x true only for bond 1-3
            idx_rot = np.arange(k, len(xyz))
            set_torsional_tmp(xyz, i, j, idx_rot, theta_rad)


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
            theta_rad = (self.get_psi(resnum) - theta) * constants.degrees_to_radians
            xyz = self.coords
            resname = self.sequence[resnum + 1]
            O_idx = templates_gl[resname].atom_names.index('O3')
            k = self._offsets[resnum + 1]
            i = k + O_idx   # index of O'x true only for bond 1-3
            
            C_idx = templates_gl[resname].atom_names.index('C3')
            j = k + C_idx   # index of C'x true only for bond 1-3    

            idx_rot = np.array(list(set(range(k, len(xyz))) - set([i])))

            set_torsional_tmp(xyz, i, j, idx_rot, theta_rad)


def _prot_builder_from_seq(sequence):
    """
    Build a protein from a template.
    Adapted from fragbuilder
    """
    names = []
    bonds_mol = []
    pept_coords, pept_at, bonds, _, _, offset = templates_aa[sequence[0]]
    names.extend(pept_at)
    bonds_mol.extend(bonds)
    offsets = [0, offset]
    for idx, aa in enumerate(sequence[1:]):
        tmp_coords, tmp_at, bonds, _, _, offset = templates_aa[aa]
        
        v3 = pept_coords[2 + offsets[idx]]  # C
        v2 = pept_coords[1 + offsets[idx]]  # CA
        v1 = pept_coords[0 + offsets[idx]]  # N
        
        connectionpoint = v3 + (v2 - v1) / mod(v2 - v1) * constants.peptide_bond_lenght
        connectionvector = tmp_coords[0] - connectionpoint

        # translate
        tmp_coords = tmp_coords - connectionvector
        
        # first rotation
        v4 = v3 - v2 + connectionpoint
        axis1 = perp_vector(tmp_coords[1], connectionpoint, v4)
        angle1 =  get_angle(tmp_coords[1], connectionpoint, v4)
        center1 = connectionpoint

        ba =  axis1 - center1
        tmp_coords = tmp_coords - center1
        tmp_coords = tmp_coords @ rotation_matrix_3d(ba, angle1)
        tmp_coords = tmp_coords + center1

        axis2 = tmp_coords[1] - connectionpoint
        axis2 = axis2 / mod(axis2) + connectionpoint
        d3 = tmp_coords[1]
        d4 = tmp_coords[2]
        if aa == 'P':
            angle2 = constants.pi + get_torsional(v3, connectionpoint, d3, d4) - 1.5707963267948966
        else:
            angle2 = constants.pi + get_torsional(v3, connectionpoint, d3, d4)
        center2 = connectionpoint
        ba =  axis2 - center2
        tmp_coords = tmp_coords - center2
        tmp_coords = tmp_coords @ rotation_matrix_3d(ba, angle2)
        tmp_coords = tmp_coords + center2
        
        names.extend(tmp_at)
        offsets.append(offsets[idx+1] + offset)
        pept_coords = np.concatenate([pept_coords, tmp_coords])

        # create a list of bonds from the template-bonds by adding the offset
        prev_offset = offsets[-3]
        last_offset = offsets[-2]
        bonds_mol.extend([(i + last_offset, j + last_offset)
                         for i, j in bonds] + [(2 + prev_offset, last_offset)])

    offsets.append(offsets[-1] + offset)
    exclusions = _exclusiones_1_3(bonds_mol)
    
    # generate a list with the names of chemical elements
    elements = []
    for i in names:
        element = i[0]
        if element in ['1', '2', '3']:
            element = i[1]
        elements.append(element)
        
    occupancies = [1.] * len(names)
    bfactors = [0.] * len(names)

    return (pept_coords,
            names,
            elements,
            occupancies,
            bfactors,
            offsets,
            exclusions)


def _builder_from_pdb(pdb, mol_type):
    """
    Auxiliary function to build a protein or glycan object from a pdb file
    """
    if mol_type == 'protein':
        templates = templates_aa
        three_to_one = three_to_one_aa
    elif mol_type == 'glycan':
        templates = templates_gl
        three_to_one = three_to_one_gl
    
    (names,
     sequence,
     mol_coords,
     occupancies,
     bfactors,
     elements) = _pdb_parser(pdb, three_to_one)

    bonds_mol = []
    _, _, bonds, _, _, offset = templates[sequence[0]]
    bonds_mol.extend(bonds)
    offsets = [0, offset]
    
    for idx, aa in enumerate(sequence[1:]):
        offset = templates[aa][-1]
        offsets.append(offsets[idx+1] + offset)
        prev_offset = offsets[-3]
        last_offset = offsets[-2]
        bonds_mol.extend([(i + last_offset, j + last_offset) 
                         for i, j in bonds] + [(2 + prev_offset, last_offset)])
    offsets.append(offsets[-1] + offset)
    exclusions = _exclusiones_1_3(bonds_mol)
    
    return (sequence,
            mol_coords,
            names,
            elements,
            occupancies,
            bfactors,
            offsets,
            exclusions)   


def _pdb_parser(filename, three_to_one):
    """
    This function is very fragile now. It's only works with files saved using
    bomeba or files that has hydrogen a single model and follows the PDB rules
    it will not work for example with files from PyMOL.
    """
    serial = []
    names = []
    #altloc = []
    resnames = []
    chainid = []
    resseq = []
    #icode = []
    xyz = []
    occupancies = []
    bfactors = []
    elements = []
    #charge = []
    for line in open(filename).readlines():
        if line[0:5] == 'ATOM ':
            name = line[12:16].strip()
            
            # this rules fix problem with  NMR pdb files, but brake reading glycans
            # turning them off until better solution
            #if name == 'H1':
            #    name = 'H'
            #if name not in ['H2', 'H3', 'OXT']:
            if True:
                serial.append(int(line[6:11]))
                names.append(name)
                #altloc.append(line[16])
                resnames.append(line[17:20])
                chainid.append(line[21])
                resseq.append(int(line[22:26]))
                #icode.append(line[26])
                xyz.append([float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54])])
                occupancies.append(float(line[54:60]))
                bfactors.append(float(line[60:66]))
                elements.append(line[76:78].strip())
                #charge.append(line[78:80])

    unique_res = sorted((set([resseq.index(i) for i in resseq])))
    sequence = ''

    for i in unique_res:
        aa = three_to_one[resnames[i]]
        sequence += aa

    return names, sequence, np.array(xyz), occupancies, bfactors, elements


def _exclusiones_1_3(bonds_mol):
    # based on the information inside bonds_mol determine the 1-3 exclusions
    # ToDo: write a not-naive version of this
    angles_mol = []
    for idx, i in enumerate(bonds_mol):
        a, b = i
        for j in bonds_mol[idx+1:]:
            c, d = j
            if (a == c and b != d):
                angles_mol.append((b, d))
            elif (a == d and b != c):
                angles_mol.append((b, c))
            elif b == c and d != a:
                angles_mol.append((a, d))
            elif b == d and c != a:
                angles_mol.append((a, c))

    exclusions = bonds_mol + angles_mol
    return set([tuple(sorted(i)) for i in exclusions])
    

def _get_rotation_indices_prot(self):
    """
    
    """
    rotation_indices = []
    for resnum in range(0, len(self)):
        d = {}
        lenght = len(self.coords)
        ###  phi  ###
        i = self._offsets[resnum]
        j = i + 1
        resname = self.sequence[resnum]
        if resname != 'P':  
            H = templates_aa[resname].atom_names.index('H')
            a = list(range(j, lenght))
            a.remove(i + H)  # H atom should not be rotated 
            idx_rot = np.array(a)  # rotation are faster if idx_rot is an array
        #else:   # XXX phi is not changed for P, should we?
        #    idx_rot = np.arrange(j, len(xyz))
        d['phi'] = i, j, idx_rot
        ###  psi  ###
        #N(i),Ca(i),C(i),N(i+1) 
        k = self._offsets[resnum] + 1
        l = k + 1
        a = list(range(self._offsets[resnum + 1], lenght))
        C = i + templates_aa[resname].atom_names.index('C')
        O = i + templates_aa[resname].atom_names.index('O')
        a.extend((C, O)) # The C and O atoms from this residue should rotate
        idx_rot = np.array(a) # rotation are faster if idx_rot is an array
        d['psi'] = k, l, idx_rot
        ###  chi  ###

        rotation_indices.append(d)
    return rotation_indices

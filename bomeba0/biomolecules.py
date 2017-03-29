import numpy as np
from .residues import aa_templates, one_to_three
from .utils import mod, dot, cross, perp_vector, get_angle, get_torsional
from .geometry import rotation_matrix_3d
from .constants import constants


class TestTube():
    """
    This is a "container" class instanciated only once (Singleton)
    """
    _instance = None
    def __new__(cls, solvent=None, temperature=298, force_field='simple_lj', *args, **kwargs):
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
        ToDo: It should be possible to compute partial energy, like without solvent or excluding molecules.
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
        print(molecules)
    
    def remove(self, name):
        """
        remove molecules from TestTube
        """
        molecules = self.molecules
        if name in molecules:
            molecules.remove(name)


class Protein:
    '''write me'''
    def __init__(self, sequence, coords=None):
        '''Initialize a new protein. The array of cartesian coordinates can be specified.
        If not, coords are built from sequence.'''
        self.sequence = sequence
        self.coords, self._names = self._checkcoords(sequence, coords)
        # add instance of Protein to TestTube automatically

    def __len__(self):
        return len(self.sequence)

    def _checkcoords(self, sequence, coords):
        """
        If coords is None return default coordinates matching sequence. 
        If not check that coords is a NumPy array of dimension (N, 3)
        
        ToDO: check that coords and sequence match
        """
        if coords is not None:
            if not isinstance(coords, np.ndarray):
                raise TypeError('Input not a ndarray')
            elif not (coords.ndim == 2 and coords.shape[1] == 3): 
                raise ValueError('Dimensions should be (N, 3)')
            else:
                return coords
        else:
            coords, names = _prot_builder(sequence)
            return coords, names

    def phi (self, resnum) :
        """
        Compute phi torsional angle

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        """
        ### FIX ME!!!
        # N_i-Ca_i-C_i-N_(i+1)
        coords = self.coords
        offset = 0
        this_offset = 10
        a = coords[offset]
        b = coords[offset + 1]
        c = coords[offset + 2]
        d = coords[offset + this_offset]
        return 180.#get_torsional(a, b, c, d) * constants.radians_to_degrees

    def psi (self, resnum) :
        """
        Compute psi torsional angle

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        """
        ### FIX ME!!!
        # Ca_i-C_i-N_(i+1)-N_(i+1)
        coords = self.coords
        offset = 0 # len(pedazo_previo)
        this_offset = 10 # len(de resnum)
        a = coords[offset - this_offset + 1]
        b = coords[offset]
        c = coords[offset + 1]
        d = coords[offset + 2]
        return 180.#get_torsional(a, b, c, d) * constants.radians_to_degrees

    def chi (self, resnum, chi_num) :
        """
        Compute chi torsional angle

        Parameters
        ----------
        resnum : int
            residue number from which to compute torsional
        chi_num : int
            number of chi, some residues have more than one chi torsional:
        """
        pass

    def dump_pdb(self, filename) :
        """
        Write a protein to a pdb file

        Parameters
        ----------
        filename : string
            name of the file without the pdb extension
        """
        # FIX THIS!!!!        
        coords = self.coords
        names = self._names
        rep_seq = ['1'] * 10
        b = ['2'] * 10
        rep_seq.extend(b)
        rep_seq_nam = ['A'] * 20


        fd = open('{}.pdb'.format(filename), 'w')
        for i in range(len(coords)):
            serial = str(i + 1)
            name = names[i]
            resname = one_to_three[rep_seq_nam[i]]
            resseq = rep_seq[i]
            element = name[0]
            if element in ['1', '2', '3']:
                element = name[1]
            line = "ATOM {:>6s}{:>4s} {:>4s} {:>5s}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           {:2s}  \n".format(serial, name, resname, resseq, *coords[i], element)
            fd.write(line)
        fd.close()


def _prot_builder(sequence):
    """
    Build a protein from a template.
    Adapted from fragbuilder
    """
    pept_coords, pept_at = aa_templates[sequence[0]]
    pept_lenght = len(pept_coords)
    for aa in sequence[1:]:
        tmp_coords, tmp_at = aa_templates[aa]
        
        v3 = pept_coords[2]  # C
        v2 = pept_coords[1]  # CA
        v1 = pept_coords[0]  # N
        
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
        angle2 = constants.pi + get_torsional(v3, connectionpoint, d3, d4) # si res_next Pro sumar 90 grados
        center2 = connectionpoint
        ba =  axis2 - center2
        tmp_coords = tmp_coords - center2
        tmp_coords = tmp_coords @ rotation_matrix_3d(ba, angle2)
        tmp_coords = tmp_coords + center2
        
        pept_at.extend(tmp_at)
        pept_coords = np.concatenate([pept_coords, tmp_coords])

    return pept_coords, pept_at

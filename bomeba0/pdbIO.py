"""
Collection of functions related to generating molecules from sequence or pdb 
files
"""

import numpy as np
import bomeba0 as bmb
from .utils import mod, perp_vector, get_angle, get_torsional
from .geometry import rotation_matrix_3d, set_torsional
from .templates.aminoacids import templates_aa, one_to_three_aa, three_to_one_aa
from .templates.glycans import templates_gl, one_to_three_gl, three_to_one_gl
from .constants import constants


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
        if sequence[0] == 'B' and idx == 0:
            v3 = pept_coords[0 + offsets[idx]]  # C
            v2 = pept_coords[2 + offsets[idx]]  # CH3
            v1 = (pept_coords[5 + offsets[idx]] + pept_coords[3 + offsets[idx]]) / 2  # HH31 / HH33
            #['C', 'O', 'CH3', 'HH31', 'HH32', 'HH33'],
        else:
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
        angle1 = get_angle(tmp_coords[1], connectionpoint, v4)
        center1 = connectionpoint

        ba = axis1 - center1
        tmp_coords = tmp_coords - center1
        tmp_coords = tmp_coords @ rotation_matrix_3d(ba, angle1)
        tmp_coords = tmp_coords + center1

        axis2 = tmp_coords[1] - connectionpoint
        axis2 = axis2 / mod(axis2) + connectionpoint
        d3 = tmp_coords[1]
        d4 = tmp_coords[2]
        if aa == 'P':
            angle2 = constants.pi + \
                get_torsional(v3, connectionpoint, d3, d4) - 1.5707963267948966
        else:
            angle2 = constants.pi + get_torsional(v3, connectionpoint, d3, d4)
        center2 = connectionpoint
        ba = axis2 - center2
        tmp_coords = tmp_coords - center2
        tmp_coords = tmp_coords @ rotation_matrix_3d(ba, angle2)
        tmp_coords = tmp_coords + center2

        names.extend(tmp_at)
        offsets.append(offsets[idx + 1] + offset)
        pept_coords = np.concatenate([pept_coords, tmp_coords])

        # create a list of bonds from the template-bonds by adding the offset
        prev_offset = offsets[-3]
        last_offset = offsets[-2]
        bonds_mol.extend([(i + last_offset, j + last_offset)
                          for i, j in bonds] + [(2 + prev_offset, last_offset)])

    offsets.append(offsets[-1] + offset)
    exclusions = _exclusions_1_3(bonds_mol)

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


def _builder_from_pdb_gl(pdb, mol_type, regularize=False, linkages=None):
    """
    Auxiliary function to build a glycan object from a pdb file
    """
    # FIXME make this function exclusive to glycans, since que already have one
    # for proteins. Also  everything related to glycans is really messy.
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
     elements,
     offsets) = _pdb_parser(pdb, three_to_one)

    if not regularize:
        exclusions = []
        return (sequence,
                mol_coords,
                names,
                elements,
                occupancies,
                bfactors,
                offsets,
                exclusions)

    else:
        bonds_mol = []
        _, _, bonds, _, _, offset = templates[sequence[0]]
        bonds_mol.extend(bonds)
        offsets = [0, offset]

        for idx, resname in enumerate(sequence[1:]):
            bonds = templates[resname][2]
            offset = templates[resname][-1]
            offsets.append(offsets[idx + 1] + offset)
            prev_offset = offsets[idx]
            last_offset = offsets[idx + 1]
            # shift index of newly added residues
            bonds_mol.extend([(i + last_offset, j + last_offset)
                              for i, j in bonds])
            # add bond between residues (not listed in the templates)
            if mol_type == 'protein':
                bonds_mol.append((2 + prev_offset, last_offset))
            elif mol_type == 'glycan':

                link = linkages[idx]
                if isinstance(link, tuple):
                    prev_offset = offsets[link[0]]
                    resname = sequence[link[0]]
                    link = link[1]
                    if link > 0:
                        O_idx = templates_gl[resname].atom_names.index('O{}'.format(link))
                        bonds_mol.append((prev_offset, last_offset + O_idx))
                        #print('1', prev_offset, last_offset + O_idx)
                    else:
                        O_idx = templates_gl[resname].atom_names.index('O{}'.format(abs(link)))
                        bonds_mol.append((prev_offset + O_idx, last_offset))
                        #print('2', prev_offset + O_idx, last_offset)
                else:
                    if link > 0:
                        O_idx = templates_gl[resname].atom_names.index('O{}'.format(link))
                        bonds_mol.append((prev_offset, last_offset + O_idx))
                        #print('3', prev_offset, last_offset + O_idx)
                    else:
                        resname = sequence[idx]
                        O_idx = templates_gl[resname].atom_names.index('O{}'.format(abs(link)))
                        bonds_mol.append((prev_offset + O_idx, last_offset))
                        #print('b', resname, prev_offset, O_idx, link)
                        #print('4', prev_offset + O_idx, last_offset)

        offsets.append(offsets[-1] + offset)
        exclusions = _exclusions_1_3(bonds_mol)

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
    #serial = []
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
            # serial.append(int(line[6:11]))
            names.append(line[12:16].strip())
            # altloc.append(line[16])
            resnames.append(line[17:20])
            chainid.append(line[21])
            resseq.append(int(line[22:26]))
            # icode.append(line[26])
            xyz.append([float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54])])
            occupancies.append(float(line[54:60]))
            bfactors.append(float(line[60:66]))
            elements.append(line[76:78].strip())
            # charge.append(line[78:80])

    unique_res = sorted((set([resseq.index(i) for i in resseq])))
    sequence = ''

    for i in unique_res:
        aa = three_to_one[resnames[i]]
        sequence += aa
    tmp = []
    offsets = []
    for idx, a in enumerate(resseq):
        if a not in tmp:
            tmp.append(a)
            offsets.append(idx)
    offsets.append(idx + 1)
    return names, sequence, np.array(xyz), occupancies, bfactors, elements, offsets


def _exclusions_1_3(bonds_mol):
    """
     Based on the information inside bonds_mol determine the 1-3 exclusions
     ToDo: write a non-naive version of this
    """
    angles_mol = []
    for idx, i in enumerate(bonds_mol):
        a, b = i
        for j in bonds_mol[idx + 1:]:
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


def _dump_pdb(self, filename, b_factors, to_file):
    """
    Write a molecule object to a pdb file

    Parameters
    ----------
    filename : string
        name of the file without the pdb extension
    b_factors : list optional
        list of values to fill the b-factor column. one value per atom
    to_file : bool
        whether to save the pdb to a file (default) or return a string
    """
    if isinstance(self, bmb.Protein):
        one_to_three = one_to_three_aa
        templates = templates_aa
    elif isinstance(self, bmb.Glycan):
        one_to_three = one_to_three_gl
        templates = templates_gl

    coords = self.coords
    names = self._names
    elements = self._elements
    sequence = self.sequence
    occupancies = self.occupancies
    offsets = self._offsets

    if b_factors is None:
        b_factors = self.bfactors
    else:
        b_factors = bfactors

    rep_seq_nam = []
    rep_seq = []
    offset_0 = offsets[0]
    for idx, aa in enumerate(sequence):
        offset_1 = offsets[idx + 1]
        lenght = offset_1 - offset_0
        offset_0 = offset_1
        seq_nam = aa * lenght
        res = [str(idx + 1)] * lenght
        rep_seq_nam.extend(seq_nam)
        rep_seq.extend(res)

    pdb_str = ''
    for i in range(len(coords)):
        serial = str(i + 1)
        name = names[i]
        if len(name) < 4:
            name = ' ' + name
        resname = one_to_three[rep_seq_nam[i]]
        resseq = rep_seq[i]
        pdb_str = pdb_str + ("ATOM {:>6s} {:<4s} {:>3s} A{:>4s}"
                             "    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}"
                             "           {:2s} \n").format(serial, name,
                                                           resname, resseq,
                                                           *coords[i],
                                                           b_factors[i],
                                                           occupancies[i],
                                                           elements[i])

    if to_file:
        fd = open('{}.pdb'.format(filename), 'w')
        fd.write(pdb_str)
        fd.close()
    else:
        return pdb_str

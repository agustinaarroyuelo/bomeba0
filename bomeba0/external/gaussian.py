import numpy as np
import os
from bomeba0.molecules import Protein


def gen_tripeptides(protein=None, gaussian=True, header=None, save_pdb=True,
                    folder=None, main_base='6-311+G(2d,p)', minor_base='3-21G'):
    """
    protein : str
        path to pdb file
    gaussian : bool
        If True a gaussian job file will be generated along the tripeptides. Defaults to True.
    header : str  
        Valid header for Gaussian job file. Only works if `gaussian` is True
    save_pdb : bool
        If True a pdb for each tripeptide will be saved to hardisk. Defaults to True.
    folder : str
        Path to the folder containing the generated files. If none provided it will use the basename of the
        protein file without extension.
    main_base : str
        valid gaussian DFT base. This will be used for the central residue's atoms.
        Defaults to `6-311+G(2d,p)`.
    minor_base : str
        valid gaussian DFT base. This will be used for the non-central residue's atoms.
        Defaults to `3-21G`.
    """
    if folder is None:
        folder = os.path.splitext(os.path.basename(protein))[0]
    prot = Protein(pdb=protein, regularize=False)
    tors = prot.get_torsionals()
    seq = prot.sequence
    for idx, aa in enumerate(seq[1:-1]):
        pept = Protein('BG{}GZ'.format(aa))
        for i in range(0, 3):
            phi = tors.iloc[idx+i].phi
            psi = tors.iloc[idx+i].psi
            if np.isfinite(phi):
                pept.set_phi(i+1, phi)
            if np.isfinite(psi):
                pept.set_psi(i+1, psi)
            if i == 2:
                for j in range(1, 6):
                    chi = tors.iloc[idx+i]['chi{}'.format(j)]
                    if np.isfinite(chi):
                        pept.set_chi(2, j, chi)

        directory = '{}/M_{}'.format(folder, idx)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if save_pdb:
            pept.dump_pdb('{}/test'.format(directory))
        if gaussian:
            _write_gaussian_file(pept, directory, header, main_base, minor_base)


def _write_gaussian_file(pept, directory, header, main_base, minor_base):
    """
    pept : protein object
    directory : str
        Path to the directory containing the generated files.
    header : str  
        Valid header for Gaussian job file. Only works if `gaussian` is True 
    main_base : str
        valid gaussian DFT base. This will be used for the central residue's atoms.
    minor_base : str
        valid gaussian DFT base. This will be used for the non-central residue's atoms.
    """
    fd = open('{}/RES.com'.format(directory), 'w')
    fd.write(header)
    for idx, xyz in enumerate(pept.coords):
        atom_type = pept._elements[idx]
        fd.write('  %s%9.3f%8.3f%8.3f\n' % (atom_type, *xyz))
    fd.write('\n')
    string = "  1   2   3   4   5   6   7   8   9  10  11  12  13 0\n 3-21G\n ****\n"
    atoms = len(pept.at_coords(2))
    for atom in range(14, atoms+14):
        string = string + ' %s ' % atom
    string = string + '0\n {}\n ****\n'.format(main_base)
    for atom in range(atoms+14, atoms+27):
        string = string + ' %s ' % atom
    string = string + '0\n {}\n ****\n\n\n'.format(minor_base)
    fd.write(string)
    fd.close()

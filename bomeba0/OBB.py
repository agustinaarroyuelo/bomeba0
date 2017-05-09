"""
Write me! 
"""
from .residues import aa_templates

def _little_boxes(prot):
        """
        Returns a list of lists, each list contains the indices of atoms
        belonging to a different box. These indices will then be used to
        retrieve the coordinates for the atoms in each box.
        See 10.1145/237170.237244 for details.

        Parameters
        ----------
        prot : protein object
        
        Returns
        ----------
        boxes : list
        """
        offsets = prot._offsets     
        
        boxes = []
        seq = prot.sequence
        lenght = len(seq) - 1
        for resnum, resname in enumerate(seq):
            resinfo = aa_templates[resname]
            offset_0 = offsets[resnum - 1]
            offset_1 = offsets[resnum]

            idx_ca = [_ + offset_1 for _ in resinfo.sc]
            idx_ca += [resinfo.atom_names.index('CA') + offset_1]
            idx_ca += [resinfo.atom_names.index('HA') + offset_1] 
    
            idx_p = [resinfo.atom_names.index('C') + offset_0]
            idx_p += [resinfo.atom_names.index('O') + offset_0]
            idx_p += [resinfo.atom_names.index('N') + offset_1] 
            idx_p += [resinfo.atom_names.index('H') + offset_1]

            if  0 < resnum < lenght: 
                boxes.append(idx_p)
                boxes.append(idx_ca)
            else:
                if resnum == 0:
                    idx_n = [resinfo.atom_names.index('N') + offset_1] 
                    idx_n += [resinfo.atom_names.index('H') + offset_1]
                    boxes.append(idx_n)
                    boxes.append(idx_ca)
                elif resnum == lenght:
                    boxes.append(idx_p)
                    boxes.append(idx_ca)
                    idx_c = [resinfo.atom_names.index('C') + offset_1]
                    idx_c += [resinfo.atom_names.index('O') + offset_1]
                    boxes.append(idx_c)
        return boxes

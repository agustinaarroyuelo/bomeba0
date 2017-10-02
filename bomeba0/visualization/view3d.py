import py3Dmol

def _view3d(mol):
    """
    draft of function to visualize a molecule embedded in a notebook
    """
    pdb_str = mol.dump_pdb('prot', to_file=False)
    view = py3Dmol.view()
    view.addModel(pdb_str,'pdb')
    view.setStyle({'cartoon':{'color':'orange'}})
    view.zoomTo()
    return view.show()


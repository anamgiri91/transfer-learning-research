import os
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data

def sdf_to_pyg_data(sdf_path: str, pKD: float = None):
    """
    Parses a SDF file to extract atomic coordinates and atomic numbers,
    and returns a PyTorch Geometric Data object suitable for SchNet.
    """
    if not os.path.exists(sdf_path):
        return None
        
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    try:
        mol = next(supplier)
    except StopIteration:
        return None
        
    if mol is None:
        return None
            
    conf = mol.GetConformer()
    
    pos = []
    z = []
    
    for i, atom in enumerate(mol.GetAtoms()):
        pos.append(list(conf.GetAtomPosition(i)))
        z.append(atom.GetAtomicNum())
        
    pos = torch.tensor(pos, dtype=torch.float32)
    z = torch.tensor(z, dtype=torch.long)
    
    data = Data(z=z, pos=pos)
    if pKD is not None:
        data.y = torch.tensor([pKD], dtype=torch.float32).view(1, 1)
        
    return data

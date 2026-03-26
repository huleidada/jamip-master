from jamip.structure import Structure,read,write
from jamip.modeling.structureFactory import StructureFactory
from jamip.structure.convert import read_structure_from_atat
import numpy as np
import pytest
import pathlib

def get_molecule(axis=[0,0,0]):
    mol = read('/public/home/slluo/Project/CsDMAPbI3/bulk/outs/DMA-HI.vasp/scf/POSCAR')
    atomi = None
    for atom in mol.atomic_positions:
        if atom.specie == 'I':
            atomi = atom
    sf = StructureFactory(mol)
    sf.del_atoms([atomi])
    return sf.structure

def get_rotated_molecule(mol, axis=[0,0,0]):
    atomn = None
    for atom in mol.atomic_positions:
        if atom.specie == 'N':
            atomn = atom.coord
    sf = StructureFactory(mol)
    if sum(axis) == 0:
        sf.rotation(atoms=np.arange(len(mol)),axis=np.array(axis), theta=np.pi/2, origin=atomn)

    return sf.structure, atomn

def get_bulk_from_sqs(path,a):
    s = read_structure_from_atat(path)
    s.lattice = s.lattice / np.mean(s.lattice_parameters[:3]) * a * 3
    s.update()
    return s

def get_bulk_from_supercell(structure, insert, mol_site=None):
    sf = StructureFactory(structure)
    sf.supercell([3,3,3])

    # replace one atom to insert atoms
    atoms = [] 
    for atom in bulk.atomic_positions:
        if atom.specie == 'Cs':
            atoms.append(atom)
            break

    if isinstance(insert, str):
        sf.substitute_atoms(atoms, specie=insert)
    else:
        coords = [atom.scale_coord for atom in atoms]
        sf.del_atoms(atoms)
        for coord in coords:
            sf.insertMolecule(insert, mol_site, coord)

    return sf.structure
   
def get_hybrid_perovskite(structure, specie, insert, mol_site=None):

    sf = StructureFactory(structure)
    atoms = []
    for atom in structure.atomic_positions:
        if atom.specie == specie:
            atoms.append(atom)

    if len(atoms) == 0:
        return sf.structure
    
    if isinstance(insert, str):
        sf.substitute_atoms(atoms, specie=insert)
    else:
        coords = [atom.scale_coord for atom in atoms]
        sf.del_atoms(atoms)
        for coord in coords:
            sf.insertMolecule(insert, mol_site, coord)


if __name__ == '__main__':

    bulk = read('/public/home/slluo/Project/CsDMAPbI3/bulk/outs/CsPbI3.vasp/scf/POSCAR')
    a = np.mean(bulk.lattice_parameters[:3])
    mol = get_molecule()
    mol, atomn = get_rotated_molecule(mol, axis=[0,0,0])

    pp = pathlib.Path('rot0')
    pp.mkdir(parents=True, exist_ok=True)

    # DMA1
    s1 = get_bulk_from_supercell(bulk, mol)
    write(s1, pp/'P-DMA.vasp')

    # Rb1
    s1 = get_bulk_from_supercell(bulk, 'Rb')
    write(s1, pp/'P-Rb.vasp')

    # Rb2
    p = pathlib.Path('sqs-2-0')
    num = 0
    for path in p.iterdir():
        num += 1
        s = get_bulk_from_sqs(path, a)
        write(s1, pp/f'P-Rb2-{num}vasp')

    # DMA2
    p = pathlib.Path('sqs-2-0')
    num = 0
    for path in p.iterdir():
        num += 1
        s = get_bulk_from_sqs(path, a)
        s1 = get_hybrid_perovskite(s, 'Rb', mol, atomn)
        write(s1, pp/f'P-DMA2-{num}.vasp')

    for i,axis in enumerate([[0,0,0],[0,0,1],[1,0,0]]):
        mol1, atomn = get_rotated_molecule(mol, axis=axis)

        pp = pathlib.Path('rot%d' %(i+1))
        pp.mkdir(parents=True, exist_ok=True)

        # sqs-27-1-1 DMA1-Rb1
        p = pathlib.Path('sqs-1-1')
        num = 0
        for path in p.iterdir():
            num += 1
            s = get_bulk_from_sqs(path, a)
            s1 = get_hybrid_perovskite(s, 'K', mol1, atomn)
            write(s1, pp/f'P-DMA1-Rb1-{num}.vasp')

        # sqs-27-1-2 DMA1-Rb2
        p = pathlib.Path('sqs-1-2')
        num = 0
        for path in p.iterdir():
            num += 1
            s = get_bulk_from_sqs(path, a)
            s1 = get_hybrid_perovskite(s, 'K', mol1, atomn)
            write(s1, pp/f'P-DMA1-Rb2-{num}.vasp')

        # sqs-27-2-1 DMA2-Rb1
        p = pathlib.Path('sqs-1-2')
        num = 0
        for path in p.iterdir():
            num += 1
            s = get_bulk_from_sqs(path, a)
            s1 = get_hybrid_perovskite(s, 'Rb', mol1, atomn)
            s1 = get_hybrid_perovskite(s, 'K', 'Rb')
            write(s1, pp/f'P-DMA2-Rb1-{num}.vasp')

        # sqs-27-2-2 DMA2-Rb2
        p = pathlib.Path('sqs-2-2')
        num = 0
        for path in p.iterdir():
            num += 1
            s = get_bulk_from_sqs(path, a)
            s1 = get_hybrid_perovskite(s, 'Rb', mol1, atomn)
            write(s1, pp/f'P-DMA2-Rb2-{num}.vasp')
    


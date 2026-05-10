# 

def phonopy2jamip(unitcell):
    from .structure import Structure
    cell = (unitcell.cell, unitcell.scaled_positions, unitcell.symbols)
    obj = Structure.from_cell(cell, comment='PHONOPY')
    return obj

def jamip2phonopy(structure):
    from phonopy.structure.atoms import PhonopyAtoms
    lattice = structure.lattice
    positions = structure.get_positions(type='direct')
    elements = structure.get_elements('symbol').tolist()
    unitcell = PhonopyAtoms(symbols=elements,cell=lattice,
                            scaled_positions=positions)
    return unitcell

def jamip2ase(structure):
    from ase.atoms import Atoms
    elements = structure.get_elements('symbol')
    positions = structure.get_positions(type='cartesian')
    pbc = False if structure.lattice is None else True
    atoms = Atoms(symbols=elements, 
                  positions=positions,
                  cell=structure.lattice,
                  pbc=pbc) 
    return atoms

def ase2jamip(structure):
    from .structure import Structure
    cell = (structure.cell.array, structure.get_scaled_positions(), structure.numbers)
    obj = Structure.from_cell(cell, comment='ASE')
    return obj

def jamip2mp(structure):
    from pymatgen.core import Structure
    elements = structure.get_elements('symbol')
    positions = structure.get_positions(type='direct')
    obj = Structure(structure.lattice, elements, positions)
    return obj

def mp2jamip(structure):
    from .structure import Structure
    elements = []
    positions = []
    for site in structure:
        elements.append(site.species_string)
        positions.append(site.frac_coords)
    cell = (structure.lattice._matrix, positions, elements)
    obj = Structure.from_cell(cell, comment='Pymatgen')
    return obj

def jamip2rdkit(structure):
    from rdkit import Chem
    from rdkit.Chem import Atom, Conformer
    
    # 初始化一个空的分子对象
    mol = Chem.RWMol()

    for symbol in structure.get_elements(type='symbol'):
        atom = Atom(symbol) 
        mol.AddAtom(atom)

    # 将构象添加到分子
    conf = Conformer(len(structure))
    for i,coord in enumerate(structure.get_positions(type='cartesian')):
        conf.SetAtomPosition(i, coord)  
    mol.AddConformer(conf)
    
    return mol.GetMol()

def jamip2db(structure):
    from jamip.db.materials import Structure
    poscar={'comment': structure.comment_line,
            'lattice': structure.lattice,
            'elements': structure.species_of_elements,
            'numbers': structure.number_of_atoms,
            'type': 'direct',
            'positions': structure.get_positions(type='direct')}
    # TODO
    #if self.isContainedConstraints and constraints != []:
    #    poscar['constraints']=constraints
    #if self.isContainedVelocities and velocities != []:
    #    poscar['velocities']=velocities
    obj = Structure().create(poscar)
    return obj
 
def read_structure_from_hdf5(path, key:str):
    from .structure import Structure
    import h5py

    obj = None
    with h5py.File(path, "r") as h5:

        if key in h5:
            cell = (h5[key]['lattice'], h5[key]['positions'], h5[key]['elements'])
            obj = Structure.from_cell(cell, comment='hdf5')
           
    return obj

def read_structure_from_atat(path):
    from .structure import Structure
    import numpy as np
    
    with open(path, 'r') as f:

        lattice = []
        for i in range(3):
            lattice.append(f.readline().split())

        matrix = []
        for i in range(3):
            matrix.append(f.readline().split())

        lattice = np.array(lattice, dtype=float) 
        matrix = np.array(matrix, dtype=float) 
        lattice = lattice @ matrix
        
        positions = []
        elements = []
        for line in f:
            row = line.split()
            positions.append(row[:3])
            elements.append(row[3])
        positions = np.array(positions, dtype=float) 
        positions = positions @ np.linalg.inv(matrix)

    s = Structure.from_cell((lattice, positions, elements))
    return s

def read_structure_from_gau(path):
    from .molecule import Molecule
    import numpy as np

    connects = []
    with open(path, 'r') as f:

        for line in f:
            if "Optimized Parameters" in line:
                for i in range(4):
                    f.readline()

                for line in f:
                    if line.strip()[0] == "!":
                        rows = line.split()[2][2:-1].split(',')
                        connects.append(np.array(rows, dtype=int))
                    else:
                        break

            elif "Input orientation" in line or "Standard orientation" in line:
                for i in range(4):
                    f.readline()

                atoms = []
                coords = []
                for line in f:
                    rows = line.split()
                    if len(rows) == 6:
                        atoms.append(rows[1])
                        coords.append(rows[3:])
                    else:
                        break

        atoms = np.array(atoms, dtype=int)
        coords = np.array(coords, dtype=float)
    mol = Molecule.from_cell((coords, atoms))
    mol.all_connections = connects 

    return mol

def read_pybel_from_gau(path):
    from openbabel import pybel

    mols = list(pybel.readfile("g16", path))
    mol = mols[-1]
    return mol
    #mol.write("mol", "opt.mol")

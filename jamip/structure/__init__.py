# coding: utf-8
# Copyright (c) JAMIP Development Team.
# Distributed under the terms of the JLU License.


#=================================================================
# This file is part of JAMIP.
#
# Copyright (C) 2017 Jilin University
#
#  JAMIP is a platform for high throughput calculation. It aims to
#  make simple to organize and run large numbers of tasks on the
#  superclusters and post-process the calculated results.
#
#  JAMIP is a useful packages integrated the interfaces for ab initio
#  programs, such as, VASP, Guassian, QE, Abinit and
#  comprehensive workflows for automatically calculating by using
#  simple parameters. Lots of methods to organize the structures
#  for high throughput calculation are provided, such as alloy,
#  heterostructures, etc.The large number of data are appended in
#  the MySQL databases for further analysis by using machine
#  learning.
#
#  JAMIP is free software. You can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published
#  by the Free sofware Foundation, either version 3 of the License,
#  or (at your option) and later version.
#
#  You should have recieved a copy of the GNU General Pulbic Lincense
#  along with JAMIP. If not, see <https://www.gnu.org/licenses/>.
#=================================================================
from .structure import Structure
from .molecule import Molecule

"module to write the structure"
def write(structure,stdout:str,ftype=None,filename=None):
    from pathlib import Path
    stdout = Path(stdout) 
    if filename is None: 
        filename = Path(stdout.name)
        stdout = stdout.parent
    else:
        filename = Path(filename)

    if ftype is None and len(filename.suffix):
        ftype = filename.suffix[1:]

    if ftype == 'vasp' or ftype == 'poscar':
        from jamip.abtools.vasp.vaspio import VaspIO
        VaspIO.write_poscar(structure,str(stdout),name=str(filename))
    elif ftype == 'cif':
        from .io import write_cif
        write_cif(structure, str(stdout/filename))
    elif ftype == 'xyz':
        from .io import write_xyz
        write_xyz(structure, str(stdout/filename))
    elif ftype == 'pdb':
        from .io import write_pdb
        write_pdb(structure, str(stdout/filename))
    # elif ftype == 'struct':
    #     from .io import write_struct
    #     write_struct(structure, str(stdout/filename))
    elif ftype == 'boltztrap':
        from .io import write_boltztrap
        write_boltztrap(structure, str(stdout/filename))
    elif ftype == 'struct':
        from jamip.abtools.wien2k.wienio import WienIO
        WienIO.write_struct(structure, str(stdout/filename))
    else:
        raise ValueError('Unsupport file type. Coming soon!')
 
"module to read the structure"

def read(name='POSCAR',ftype=None, standard=True):
    """
    :param name: the input structure file;
    :return: object of Structure;
    """
    from .crystal import Read

    structure = Read(name, ftype).getStructure()

    if 'lattice' in structure:
        obj = Structure()
        obj.comment_line = structure.get('comment', 'JAMIP')
        obj.lattice = structure['lattice']
        obj.atomic_coord_format = structure['type']
        if 'numbers' in structure:
            obj.species_of_elements = structure['elements']
            obj.number_of_atoms = structure['numbers']
            obj.atomic_positions = structure['positions']
        elif standard is True:
            species, numbers, positions = obj.elements2species(structure['elements'], structure['positions'])
            obj.species_of_elements = species
            obj.number_of_atoms = numbers
            obj.atomic_positions = positions
        else:
            species, numbers = obj.elements2species(structure['elements'])
            obj.species_of_elements = species
            obj.number_of_atoms = numbers
            obj.atomic_positions = structure['positions']

        if 'constraints' in structure and len(structure['constraints']):
            obj.select_dynamic = structure['constraints']
        if 'velocities' in structure and len(structure['velocities']):
            obj.initial_velocity = structure['velocities'] 

    else:

        obj = Molecule()
        obj.comment_line = structure.get('comment', 'JAMIP')
        if 'numbers' in structure:
            obj.species_of_elements = structure['elements']
            obj.number_of_atoms = structure['numbers']
            obj.atomic_positions = structure['positions']
        elif standard is True:
            species, numbers, positions = obj.elements2species(structure['elements'], structure['positions'])
            obj.species_of_elements = species
            obj.number_of_atoms = numbers
            obj.atomic_positions = positions
        else:
            species, numbers = obj.elements2species(structure['elements'])
            obj.species_of_elements = species
            obj.number_of_atoms = numbers
            obj.atomic_positions = structure['positions']
        if 'connections' in structure:
            obj.connections = structure['connections']
        if 'mol_lattice' in structure:
            obj.lattice = structure['mol_lattice']

    return obj

def reads(name='POSCAR',ftype=None):
    """
    :param name: the input structure file;
    :return: object of Structure;
    """
    from .crystal import Read

    structures = []
    r = Read(name, ftype, multi=True)

    while True:
        structure = r.getStructure()
 
        if 'lattice' in structure:
            obj = Structure()
            obj.comment_line = structure.get('comment', 'JAMIP')
            obj.lattice = structure['lattice']
            obj.atomic_coord_format = structure['type']
            obj.species_of_elements = structure['elements']
            obj.number_of_atoms = structure['numbers']
            obj.atomic_positions = structure['positions']
 
            if 'constraints' in structure and len(structure['constraints']):
                obj.select_dynamic = structure['constraints']
            if 'velocities' in structure and len(structure['velocities']):
                obj.initial_velocity = structure['velocities'] 
 
        else:
 
            obj = Molecule()
            obj.comment_line = structure.get('comment', 'JAMIP')
            obj.species_of_elements = structure['elements']
            obj.number_of_atoms = structure['numbers']
            obj.atomic_positions = structure['positions']
            if 'connections' in structure:
                obj.connections = structure['connections']

        structures.append(obj)
        if r.pointer is None:
            break

    return structures

from jamip.structure import Structure,read,write
from jamip.modeling.structureFactory import StructureFactory
from jamip.modeling.interfaceFactory import InterfaceFactory
import numpy as np
import pytest
import pathlib



class Test_sf:

    def test_lattice(self,request):

        cif = request.config.rootdir / 'testfile/structure/NaCl-cubic.vasp'

        s0 = read(cif)
        sf = StructureFactory(s0)
        sf.scale([1,1,1])
        sf.redefine([[2,0,0],[0,2,0],[0,0,2]])
        sf.standardize(symprec=1e-5)
        sf.primitive(symprec=1e-3)
        sf.conventional()
        sf.supercell([2,2,2])
        sf.rotation(atoms=[0], axis=[0,0,1], theta=30)
        sf.translation(atoms=[0], direction=[0,0,0.5])

    def test_atoms(self,request):

        cif = request.config.rootdir / 'testfile/structure/NaCl-cubic.vasp'
        s0 = read(cif)
        sf = StructureFactory(s0)
        sf.add_atoms([0.25,0.25,0], 'K')
        sf.del_atoms([[0.25,0.25,0]])
        atoms = []
        for atom in s0.atomic_positions:
            if atom.specie == 'Na':
                atoms.append(atom)
        sf.substitute_atoms(atoms, 'K')

    def test_layers(self,request):

        cif = request.config.rootdir / 'testfile/structure/MoS2.vasp'
        s0 = read(cif)
        sf = StructureFactory(s0)
        sf.vacuum(direction=[0,0,1])
        sf.center(direction=[0,0,1])
        # sf.joint()
        sf.surface(hkl=[1,1,0])

    def test_atom_properties(self,request):

        cif = request.config.rootdir / 'testfile/structure/MoS2.vasp'
        s0 = read(cif)
        sf = StructureFactory(s0)
        # sf.magnetism_order('Mo', 'magmom', 1)
        # sf.constraint(atoms, 'magmom', 1)
        sf.initializeVelocityDistribution(temperature=300)
        sf.perturb(cutoff=0.1)

    def test_unit(self,request):

        cif = request.config.rootdir / 'testfile/structure/MoS2.vasp'
        s0 = read(cif)
        sf = StructureFactory(s0)
        sf.getUnit(unit='A')
        sf.removeUnit(unit='A')
        sf.addUnit(unit='A')
        sf.insertMolecule(molecule='H2O', position=[0,0,0])


if __name__ == '__main__':
    # pytest.main(['-vs'])

    rootdir = pathlib.Path('/public/home/slluo/test/jamip-test/')
    cif = rootdir / 'testfile/structure/MoS2.vasp'
    s0 = read(cif)
    sf = StructureFactory(s0)
    # sf.scale([1,1,1])
    # sf.redefine([[2,0,0],[0,2,0],[0,0,2]])
    # sf.standardize(symprec=1e-5)
    # sf.primitive(symprec=1e-3)
    # sf.conventional()
    # sf.supercell([2,2,2])
    # sf.rotation(atoms=[0], axis=[0,0,1], theta=30)
    # sf.translation(atoms=[0], direction=[0,0,0.5])
    # sf.surface(hkl=[1,1,0])    
    # sf.vacuum(direction=[0,0,1])
    # sf.center(direction=[0,0,1])
    # sf.add_atoms([0.25,0.25,0], 'K')
    # sf.del_atoms([[0.25,0.25,0]])
    # atoms = []
    # for atom in s0.atomic_positions:
    #     if atom.specie == 'Na':
    #         atoms.append(atom)
    # sf.substitute_atoms(atoms, 'K')
    # sf.initializeVelocityDistribution(temperature=300)
    # sf.perturb(cutoff=0.1)
    sf.magnetism_order({'Mo':3, 'S':0})
    sf.constraint([0,1], freezes=False)
    sf.constraint([0,1], freezes=[True, True,False], inverse=True)
    for atom in sf.structure.atomic_positions:
        print(atom)


from jamip.structure import Structure,read,write
from jamip.modeling.structureFactory import StructureFactory
from jamip.modeling.interfaceFactory import InterfaceFactory
from jamip.modeling.structureAnalyser import StrcutureMath
import numpy as np
import pytest
import pathlib



class Test_sfg:

    def xtest_lattice(self,request):

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

    def xtest_atoms(self,request):

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

    def xtest_layers(self,request):

        cif = request.config.rootdir / 'testfile/structure/MoS2.vasp'
        s0 = read(cif)
        sf = StructureFactory(s0)
        sf.vacuum(direction=[0,0,1])
        sf.center(direction=[0,0,1])
        # sf.joint()
        sf.surface(hkl=[1,1,0])

def part_replace(structure, specie='O'):

        sf = StructureFactory(structure)
        atoms = []
        for atom in structure.atomic_positions:
            if atom.specie == specie:
                atoms.append(atom)
        sf.substitute_atoms(atoms, 'S')
        return sf.structure


if __name__ == '__main__':
    # pytest.main(['-vs'])
    # from spglib import spglib
    # from spgrep import get_spacegroup_irreps
    # import spgrep
    import time

    
    rootdir = pathlib.Path('/public/home/slluo/test/jamip-test/')
    cif = rootdir / 'testfile/structure/MoS2.vasp'
    cif = '160_1_2.cif'
    s1 = read(cif)
    # s1 = part_replace(s1, specie='O')
    
    cif = '160_2_2.cif'
    s2 = read(cif)
    print(s1.get_formula())
    print(s2.get_formula())

    # exit()

    # cif = rootdir / 'testfile/structure/WS2.vasp'
    # s2 = read(cif)

    # from jamip.structure.symmetry2 import EqualTools

    # t0 = time.time()
    # eq = EqualTools.from_lattice(s1.lattice)
    # symbol, shift, indices = eq.is_structure_equal(s1, s2)
    # print(symbol, shift, indices)    
    # t1 = time.time()
    # print(t1-t0)

    
    t0 = time.time()
    sm = StrcutureMath(s1, s2, attempt_supercell=True, allow_subset=True)
    print(sm.fit())
    # print(sm.fit_anonymous())
    t1 = time.time()
    print(t1-t0)

    # # cif = rootdir / 'testfile/structure/MoS2.vasp'
    # cif = rootdir / 'testfile/structure/LiCoO2.cif'
    # s0 = read(cif)

    # dataset = spglib.get_symmetry_dataset(s0.to_cell(), symprec=1e-2)
    # # print(dataset[''])
    # # print(dataset['equivalent_atoms'])
    # print(dataset)

    # rotations = np.array(dataset['rotations'])
    # translations = np.array(dataset['translations'])
    # def gene_layer_with_equal_atoms(dataset):
    
    # sf = StructureFactory(s0)
    # # sf.redefine([[1,1,0],[-1,2,0],[0,0,1]])
    # sf.redefine([[0,1,0],[-1,1,0],[0,0,2]])
    # dataset = spglib.get_symmetry_dataset(sf.structure.to_cell(), symprec=1e-2)
    # print(dataset)

    # lattice, positions, numbers = sf.structure.to_cell()
    # kpoint = [0.5,0,0.5]
    # kpoint = [1/3,2/3,0]
    # print(positions.shape)

    # irreps, rotations, translations, mapping = get_spacegroup_irreps(lattice, positions, numbers, kpoint, symprec=1e-2)
    # print(np.array(irreps))
    # # print(numbers)
    # print(mapping)
    # irreps, mapping = spgrep.get_spacegroup_irreps_from_primitive_symmetry(rotations, translations, kpoint)
    # generators = spgrep.pointgroup.get_pointgroup_chain_generators(rotations)
    # print(generators)
    # print([rotations[i] for i in generators])
    # irreps = spgrep.get_crystallographic_pointgroup_irreps_from_symmetry(rotations)
    # # lattice, positions, numbers, kpoint, symprec=1e-2)
    # print(irreps)
    # print(mapping)
    # print(rotations.shape)
    # print(translations.shape)

    


    



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


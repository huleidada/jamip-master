# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
from jamip.structure import Structure
from jamip.structure.atom import Atom
from copy import deepcopy
from .check import *


class StructureFactory(object):
    
    def __init__(self, structure, **kwargs):
        """
        Arguments:
            structure: structure's object.
            
            kwargs:
                isCloneFullInfo (default=False): whether to clone all information of structure.
        """
        self._raw_structure=structure
        self._structure=deepcopy(structure)
            
    @property
    def raw_structure(self):
        """
        raw structure.
        """
        return self._raw_structure
        
    @property
    def structure(self):
        """
        structure object after operation.
        """
        return self._structure
    
    def scale(self, value):
        """
        scale the lattice vector by given direction.
        
        Arguments:
            value: coefficient of zoom for lattice parameters along three axes. i.e. [0.9, 1, 1], [0.9, 0.9, 1], [0.9, 0.9, 0.9]
            
        Returns:
            structureFactory's object.
        """
        structure=self._structure
        
        # check
        if np.issubdtype(type(value), np.number):
            assert value > 0, "Scale value %s <= 0!" %value
        else:
            assert len(value) == 3, "Scale list length != 3" 
            assert min(value) > 0, "Scale value %s <= 0!" %min(value)
            value = np.array(value, float)
            
        lattice = structure.lattice
        structure.lattice = (lattice.T*value).T

        return self
    
    def add_atoms(self, positions, species, position_format='direct', **kwargs):
        """
        add atoms to structure.
        
        Arguments:
            positions:  
                collection of atom's object or formated string. i.e. [atom0, atom1, atom2,...] 
                    ['Na', 0.1, 0.0, 0.0, 'Direct']
                    ['Na', 0.1, 0.0, 0.0]
                    ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                    
                    contain species information:
                    ['Na1+', 0.1, 0.0, 0.0, 'Direct']
                    ['Na1+', 0.1, 0.0, 0.0]
                    ['Na1+', 5.234, 0.0, 0.0, 'Cartesian']
                    
            kwargs:
                isNormalizingCoordinate (default=True): whether to remove the periodic boundary condition, 
                    ensure the value of atomic coordinate is between 0 and 1 (i.e. 1.3 -> 0.3).
                precision (default=1e-3): used to determine whether the two atoms are overlapped. Note that, 
                        to determine whether this atom is in collection by comparing its distance 
                        from other atoms.
        
        Returns:
            structureFactory's object.
        """
        # remove atomic translation periodicity
        
        structure=self._structure

        positions = formated_positions(positions)
        if position_format.lower() == 'cartesian':
            positions = positions @ np.linalg.inv(structure.lattice)
        species = formated_elements(species, len(positions))
        for specie, position in zip(species, positions):
            atom = Atom(specie, position, structure=structure)
            structure.add_atom(atom)
        structure.update()
        
        return self
    
    def del_atoms(self, atoms, **kwargs):
        """
        delete atoms from structure.
        
        Arguments:
            atoms: collection of atom's formated atom or object. i.e. [atom0, atom1, atom2,...] 
                    ['Na', 0.1, 0.0, 0.0, 'Direct']
                    ['Na', 0.1, 0.0, 0.0]
                    ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                    
                    contain species information:
                    ['Na1+', 0.1, 0.0, 0.0, 'Direct']
                    ['Na1+', 0.1, 0.0, 0.0]
                    ['Na1+', 5.234, 0.0, 0.0, 'Cartesian']
                    
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
        
        Returns:
            structureFactory's object.
        """
        structure=self._structure
        
        for atom in list(atoms):
            specie=None
            if isinstance(atom, Atom):
                position = atom.scale_coord
                specie = atom.specie
            else:
                position = formated_vector(atom) 
            structure.del_atom(position, specie=specie, is_exist=True)
        structure.update()

        return self
    
    def substitute_atoms(self, atoms, species, **kwargs):
        """
        delete atoms from structure.
        
        Arguments:
            atoms: collection of atom's formated atom or object. i.e. [atom0, atom1, atom2,...] 
                    ['Na', 0.1, 0.0, 0.0, 'Direct']
                    ['Na', 0.1, 0.0, 0.0]
                    ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                    
                    contain species information:
                    ['Na1+', 0.1, 0.0, 0.0, 'Direct']
                    ['Na1+', 0.1, 0.0, 0.0]
                    ['Na1+', 5.234, 0.0, 0.0, 'Cartesian']
                    
            symbol_of_elements: element's symbol. If replacing by an element for all atom, you can only specify the a element' symbol.
                i.e. 'Na', ['Na', 'Na', 'Na']
                    
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
        
        Returns:
            structureFactory's object.
        """
        structure=self._structure
        species = formated_elements(species, len(atoms))
        
        for specie, atom in zip(species, atoms):
            if isinstance(atom, Atom):
                position = atom.scale_coord
            else:
                position = formated_vector(atom) 
            structure.substitute_atom(position, specie=specie, is_exist=True)
        structure.update()

        return self
    
    def center(self, direction, dtype_of_move='position', **kwargs):
        """
        move atoms to center by given direction.
        
        arguments:
            direction: direction vector. The valid format is [1, 0, 0], [0, 1, 0], [0, 0, 1].
            dtype_of_move (default='position'): type of move. 
                'mass': moving by center of mass.
                'position': moving by boundary of position.
            isUpdatedInfo (default=False): whether to update the composition and symmetry information (include the site, operation, wyckoffSite, spacegroup).
            
        Returns:
            structureFactory's object.
        """
        structure = self._structure
        direction = formated_vector(direction)
        positions = structure.get_positions(type='direct')
        masses = np.array([atom.elementinfo.mass for atom in structure.atomic_positions])
        vector = [0,0,0]

        for i,value in enumerate(direction):
            if value == 0: 
                vector[i] = 0

            elif dtype_of_move.lower().startswith('m'): # mass
                mr = masses * positions[:,i]
                mass_center = sum(mr)/sum(masses)
                vector[i] = 0.5 - mass_center

            elif dtype_of_move.lower().startswith('p'): # position
                position_center = (positions[:,i].max()+positions[:,i].min()) / 2
                vector[i] = 0.5 - position_center

            else:
                raise ValueError('unknown type')

        # shift atoms %
        for atom in structure.atomic_positions:
            atom.scale_coord = atom.scale_coord + vector

        #structure.update()
        self._structure=structure
        return self
    
    def vacuum(self, direction, axis='z', **kwargs):
        """
        add vacuum along a direction.
        
        arguments:
            direction: direction vector to add the vacuum along lattice vector(a/b/c). The valid format is :
                [0.1, 0, 0, 'Direct']
                [0.1, 0, 0] (for Direct, can not be specify)
                [5.234, 0, 0, 'Cartesian'] (for Cartesian, must be given)
            isUpdatedInfo (default=False): whether to update the composition and symmetry information (include the site, operation, wyckoffSite, spacegroup).
            
            kwargs:
                isCenter (default=True): whether to centralize for all atoms in structure.
                distance: moving distance for all atoms in the structure along the given direction (unit: Angstrom). i.e. 1.0 
                    Note that distance don't beyond the lattice after vacuuming (<= direction).
                #isConstraint (default=False):True/False .
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
        
        Returns:
            structureFactory's object.
        """
        structure=self._structure
        lattice=structure.lattice
        lattice_parameters=structure.lattice_parameters
        #print(direction)
        #print(lattice_parameters)
        
        if is_list_or_array(direction):
            if len(direction) == 3:
                scales = formated_vector(direction) * lattice_parameters[:3]
            elif len(direction) == 4:
                if direction[3].lower() == 'direct':
                    scales = formated_vector(direction[:3]) * lattice_parameters[:3]
                if direction[3].lower() == 'cartesian':
                    scales = formated_vector(direction[:3])
                else:
                    raise

        elif isinstance(direction, (int, float)):
            scales = [0, 0, 0]
            for i,j in enumerate('xyz'):
                if j in axis:
                    scales[i] = direction 
        else:
            raise

        # add vacuum layer
        for i in range(3):
            if scales[i] != 0: # vacuum direction
                scale = scales[i]/lattice_parameters[i] # part of vacuum
                lattice[i] = lattice[i]*(1+scale)

        cart_positions = structure.get_positions(type='cartesian')
        frac_positions = cart_positions @ np.linalg.inv(lattice)

        structure.lattice = lattice
        for i,atom in enumerate(structure.atomic_positions):
            atom.lattice = structure._cell
            atom.scale_coord = frac_positions[i]

        structure.update()

        return self
    
    def magnetism_order(self, element_magmoms):
        """
        At present, only consider FM configuration. Other magnetic configuration need to set the atomic magnetism by hand.
        
        Arguments
            element_magmoms: dictionary of element's symbol and its magnetic moment. The valid formation:
                {'Fe':5,
                 'Cr':3,
                ...}
            
        Returns:
            structureFactory's object.
        """
        structure=self._structure
        for atom in structure.atomic_positions:
            if atom.specie in element_magmoms:
                atom.magmom = element_magmoms[atom.specie]
            else:
                atom.magmom=0.0
        
        return self
    
    def constraint(self, atoms, freezes, inverse=False, **kwargs):
        """
        selected dynamics. assign the constraint information to given atoms. Meanwhile, the constraint of remainder atoms 
            are set to the default [False, False, False] if don't give the value of 'constraint_of_remainder'.
        
        Arguments:
            atoms: collection of atom contain constraint information. The valid formation:
                ['Na', 0.1, 0.0, 0.0, True, True, False],
                ['Na', 0.1, 0.0, 0.0, 'Direct, True, True, False],
                ['Na', 5.234, 0.0, 0.0, 'Cartesian', True, True, False],                
                contain species information:
                ['Na1+', 0.1, 0.0, 0.0, 'Direct', True, True, False]
                ['Na1+', 0.1, 0.0, 0.0, True, True, False]
                ['Na1+', 5.234, 0.0, 0.0, 'Cartesian', True, True, False]                
                [atom1, True, True, True]]
    
            freezes: whether to freeze the atom. The valid formation:
                [True, True, True]
                [True, False, True]

            inverse: whether to inverse the atom information.
            
        Returns:
            structureFactory's object.
        """        
        structure=self._structure

        # get atom indices % 
        indices = []
        for atom in atoms:
            if isinstance(atom, (int, np.int32, np.int64)):
                indices.append(atom)
            elif isinstance(atom, Atom):
                index = structure.get_atom(atom.scale_coord, specie=atom.specie)
                indices.extend(index)
            else:
                index = structure.get_atom(atom)
                indices.extend(index)

        if inverse:
            indices = np.delete(np.arange(len(structure)), indices)

        if isinstance(freezes, bool):
            freezes = [freezes] * len(indices)
        elif len(freezes) == len(indices):
            pass
        elif len(freezes) == 3:
            freezes = [freezes] * len(indices)
        else:
            raise ValueError('unknown freezes')

        # update constraints
        for i,j in enumerate(indices):
            atom = structure.atomic_positions[j]
            atom.freeze = freezes[i]
        structure._frozen = True

        return self
    
    def redefine(self, operator_matrix, tolerance=1e-3):
        """
        redefine lattcie cell: C'=C x M.
        
        Arguments:
            operator_matrix: operator matrix (M). The valid formation:
                [[0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 0]]
                Note that the component of M should be integer. And the volume of M is an integer greater than 0.
                   
        Returns:
            structureFactory's object.
        """        
        structure=self._structure
        matrix = formated_matrix(operator_matrix)
        elements = structure.get_elements()
        positions = structure.get_positions(type='direct')
        positions = positions - np.floor(positions)
        
        # suerpcell
        def get_range(array):
            dmax = np.abs(array).sum()
            return int(np.ceil(dmax))
        
        dim1 = get_range(matrix[:,0])
        dim2 = get_range(matrix[:,1])
        dim3 = get_range(matrix[:,2])
        grid = np.mgrid[-dim1:dim1+1, -dim2:dim2+1, -dim3:dim3+1].reshape(3,-1).T # x,3
        grid_positions = positions[None,:,:] + grid[:,None,:] # ngrid, natom, 3
        #cart_positions = normalized(grid_positions @ lattice @ np.linalg.inv(matrix.T@lattice))
        cart_positions = grid_positions @ np.linalg.inv(matrix)
        cart_positions = np.around(cart_positions,8)
        cart_positions -= np.floor(cart_positions)
        #indices = np.where((np.max(cart_positions, axis=2) < (1+1e-8) ) & (np.min(cart_positions, axis=2) > 1e-8))
        indices = np.where((np.max(cart_positions, axis=2) > -1))

        new_elements = []
        new_positions = []
 
        for i,j in zip(*indices):
            coord = cart_positions[i,j]
            new_elements.append(elements[j])
            new_positions.append(coord)

        #print(new_positions)
        duplicates = remove_duplicates_with_tolerance(new_positions, tolerance)
        new_positions = np.array(new_positions)[~duplicates]
        new_elements = np.array(new_elements)[~duplicates]
        ndim = np.abs(np.around(np.linalg.det(matrix)).astype(int))
        # print(matrix)
        # print(np.abs(np.linalg.det(new_lattice)), np.linalg.det(structure.lattice))
        # print(ndim)

        if abs(np.abs(np.linalg.det(matrix)) - len(new_elements)/len(elements)) > 1e-2:
            raise RuntimeError("search atoms failed. ndim is %s, need %d atom, find %d atom" %(ndim, ndim*len(elements), len(new_elements)))

        new_lattice = matrix @ structure.lattice
        self._structure = Structure.from_cell((new_lattice, new_positions, new_elements))

        return self
    
    def standardize(self, symprec=1e-5, **kwargs):
        """
        convert to standardized structure. 
        Note that if not specify the hall number, always the first one (the smallest serial number corresponding to 
        the space-group-type in list of space groups (Seto's web site)) among possible choices and settings is chosen as default.
        
        Arguments:
            symprec (default=1e-5): precision when to find the symmetry.
            
            kwargs:                
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
                hall_number (default=0): hall number.
                
        Returns:
            structure's self (standardized).
        """
        import spglib
        
        cell=self._structure.to_cell()     
        dataset=spglib.get_symmetry_dataset(cell, symprec=symprec, **kwargs)
        lattice_std = dataset['std_lattice']
        positions_std = dataset['std_positions']
        elements_std = dataset['std_types']
        
        # TODO: inherit atom properties from old structure
        structure = Structure.from_cell((lattice_std, positions_std, elements_std))
        self._structure=structure
        return self
    
    def primitive(self, symprec=1e-5, **kwargs):
        """
        primitive structure.
        
        Arguments:
            symprec (default=1e-5, symmetry tolerance): distance tolerance in Cartesian coordinates to find crystal symmetry.
        
        Returns:
            structureFactory's object.
        """
        import spglib
        
        cell=self._structure.to_cell()
        cell_new=spglib.find_primitive(cell, symprec=symprec, **kwargs)
        if cell_new is None: raise ValueError('The search is filed')        
        self._structure=Structure.from_cell(cell_new)

        return self
    
    def conventional(self, symprec=1e-5, **kwargs):
        """
        conventional structure.
        
        Arguments:
            symprec (default=1e-5, symmetry tolerance): distance tolerance in Cartesian coordinates to find crystal symmetry.buj
        
        Returns:
            structureFactory's object.
        """
        import spglib
        
        cell=self._structure.to_cell() 
        cell_new=spglib.standardize_cell(cell, symprec=symprec, **kwargs)
        if cell_new is None: raise ValueError('The search is filed')
        self._structure=Structure.from_cell(cell_new)
        
        return self
            
    def niggli_reduce(self, eps=1e-5):
        """
        Niggli reduction.
        
        Arguments:
            eps (default=1e-5): tolerance parameter, but unlike symprec the unit is not a length.
                This is used to check if difference of norms of two basis vectors 
                is close to zero or not and if two basis vectors are orthogonal
                by the value of dot product being close to zero or not. The detail
                is shown at https://atztogo.github.io/niggli/.
        
        Returns:
            Niggli reduction.
        """
        import spglib
        structure = self._structure
        niggli_lattice=spglib.niggli_reduce(structure.lattice, eps=eps)
        structure.lattice = niggli_lattice
        structure.update()
        return self
    
    def delaunay_reduce(self, eps=1e-5):
        """
        Delaunay reduction.
        
        Arguments:
            eps (default=1e-5): tolerance parameter, see niggliReduce.
        """
        import spglib        
        structure = self._structure
        delaunay_lattice=spglib.delaunay_reduce(self.structure.lattice, eps=eps)
        structure.lattice = delaunay_lattice
        structure.update()
        return self
    
    def supercell(self, dim, **kwargs):
        """
        supercell structure.
        
        Arguments:
            dim: size of supercell. i.e. [2, 2, 2] (integral)

        Returns:
            structureFactory's object.
        """
        from copy import deepcopy
        
        structure=self._structure
        
        # check
        assert len(dim) == 3, 'invalid dim'
        dim=np.array(dim, dtype=int)
        assert min(dim) >= 1, 'invalid value in dim'
        
        # lattice
        lattice = structure.lattice
        for i in range(0, len(dim)):
            lattice[i] *= dim[i]

        new_structure = Structure()
        new_structure.clear()
        new_structure.lattice = lattice
        
        # atoms
        grid = np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]].reshape(3, -1).T
        for atom in structure.atomic_positions:
            atom.lattice = lattice
            coords = (atom.scale_coord+grid)/dim
            for coord in coords:                
                new_atom=deepcopy(atom)
                new_atom.scale_coord=coord
                new_structure.add_atom(new_atom)

        new_structure.update()
        self._structure=new_structure
        return self
    
    def joint(self, jointed_structure, direction, **kwargs):
        """
        joint two structures along given direction. Note that the size of the section perpendicular to the splicing direction 
            is equal to that of the structure, not the jointed structure.
        
        Arguments:
            jointed_structure: structure needed to joint.
            direction: direction vector to add the vacuum along lattice vector(a/b/c). The valid format is :
                [1, 0, 0, 'Direct'] # right of a
                [-1, 0, 0, 'Direct'] # left of a
            isUpdatedInfo (default=False): whether to update the composition and symmetry information (include the site, operation, wyckoffSite, spacegroup).
            
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
        
        Returns:
            structureFactory's object.    
        """
        from ..utils.check import check_formated_position_only_direct
        from ..materials.atom import Atom
        
        structure=self._structure
        
        # check lattice parameters
        if not check_formated_position_only_direct(direction): raise ValueError('unkown direction')
        if not((direction.count(0) == 2) and ((direction.count(1) == 1) or (direction.count(-1) == 1))): raise ValueError('unkown direction')
        
        atoms=structure.atoms
        atoms_in_jointed_structure=jointed_structure.atoms
        
        lattice_parameters=structure.lattice_parameters
        lattice_parameters_in_jointed_structure=jointed_structure.lattice_parameters
        
        index_of_jointed_direction= direction.index(1) if 1 in direction else direction.index(-1)
        scale=(lattice_parameters[index_of_jointed_direction]+lattice_parameters_in_jointed_structure[index_of_jointed_direction])/lattice_parameters[index_of_jointed_direction]
        structure.lattice[index_of_jointed_direction] *= scale
        
        for atom in atoms:
            if 1 in direction: # add right side
                atom.position[index_of_jointed_direction]=atom.position[index_of_jointed_direction]/scale
            else: # add left side
                atom.position[index_of_jointed_direction]=(scale-1+atom.position[index_of_jointed_direction])/scale
        for atom2 in atoms_in_jointed_structure:
            formated_atom2=atom2.to_formated_atom()
            if 1 in direction: # add right side
                formated_atom2[index_of_jointed_direction+1]=(1+formated_atom2[index_of_jointed_direction+1]*(scale-1))/scale
            else: # add left side
                formated_atom2[index_of_jointed_direction+1]=formated_atom2[index_of_jointed_direction+1]*(scale-1)/scale
            structure.add_atom(Atom().create(formated_atom=formated_atom2))
        
        self._structure=structure
        return self
    
    def cut(self, lattce_surface, **kwargs):
        """
        cut along a lattice surface.
        """
        pass
    
    def mirror(self, atoms, mirror_plane, **kwargs):
        """
        mirror given atoms along
        """
        pass
    
    def alloy(self):
        """
        """
        pass

    def find_minimal_orthogonal_lattice_vectors(self, a, b, bound:int=20, atol=1e-8, **kwargs):
        """
        Given basis vectors a, b in R^2, find integer coefficients (x1, y1, x2, y2)
        such that m = x1*a + y1*b and n = x2*a + y2*b are orthogonal (m·n = 0),
        and S = |x1*y2 - x2*y1| * |a x b| is minimized (S > 0).
    
        Returns: (S_min, (x1, y1, x2, y2)) or (None, None) if no solution exists
        """
        from itertools import product
        import math
        # Gram matrix and cross product
        A, B, C = np.dot(a, a), np.dot(a, b), np.dot(b, b)
        
        # Search bounds
        min_area, best = float('inf'), None
        
        for i, j, k in product(range(-bound, bound+1), repeat=3):
            if (i, j, k) == (0, 0, 0) or not np.isclose(i*A + j*B + k*C, 0, atol=atol):
                continue

            D = j*j - 4*i*k
            if D < 0:
                continue
         
            sqrt_D = int(math.isqrt(D)) if D == math.isqrt(D)**2 else -1
            if sqrt_D < 0:
                continue
         
            for sign in (-1, 1):
                xy = (j + sign * sqrt_D)
                if xy % 2 != 0:
                    continue
                xy //= 2
         
                # 寻找x1, y2满足 x1*y2 = xy
                for x1 in range(-bound, bound+1):
                    if x1 == 0 or (i != 0 and (x1 == 0 or i % x1 != 0)):
                        continue
         
                    x2 = i // x1 if i != 0 else 0
         
                    for y2 in range(-bound, bound+1):
                        if y2 == 0 or x1*y2 != xy or (k != 0 and (y2 == 0 or k % y2 != 0)):
                            continue
         
                        y1 = k // y2 if k != 0 else 0
         
                        if x1*y2 + x2*y1 != j:
                            continue
         
                        det = x1*y2 - x2*y1
                        if det == 0:
                            continue
         
                        # 验证正交性
                        m, n = x1*a + y1*b, x2*a + y2*b
                        if not math.isclose(np.dot(m, n), 0, abs_tol=1e-8):
                            continue

                        area = abs(det)
                        if area < min_area:
                            min_area, best = area, (x1, y1, x2, y2)            
            
        return (min_area, best) if best else (None, None)

    @classmethod
    def _surface(cls, structure, hkl, maxindex=None):
        from .interfaceFactory import vec_angle

        # remove zero 
        hkl_0 = np.array(hkl)
        hkl_0[np.where(hkl==0)] = 1
        m = np.lcm.reduce(hkl_0)
        sign = np.sign(np.prod(hkl_0))

        # base uvw vectors
        a_uvw = np.zeros(3)
        b_uvw = np.zeros(3)
        if np.count_nonzero(hkl) == 3:
            a_uvw = np.array([-m/hkl[0], m/hkl[1], 0], dtype=int)
            b_uvw = np.array([-m/hkl[0], 0, m/hkl[2]], dtype=int)
        elif np.count_nonzero(hkl) == 2:
            a_uvw[np.where(hkl!=0)] = np.array([-m,m]) / hkl[np.where(hkl!=0)]
            b_uvw[np.where(hkl==0)] = 1
        elif np.count_nonzero(hkl) == 1:
            a_uvw[np.argwhere(hkl==0)[0]] = 1
            b_uvw[np.argwhere(hkl==0)[1]] = 1
            m = 1
        else:
            raise ValueError('hkl cannot be all zeros')

        if maxindex == None:
            maxindex = int(np.max([np.abs(a_uvw), np.abs(b_uvw), np.abs(hkl)]))
        planenormal = sign * np.cross(np.dot(a_uvw, structure.lattice),
                                   np.dot(b_uvw, structure.lattice))

        def gen_grid(n):
            grid = np.mgrid[-n:n+1,-n:n+1,-n:n+1].T.reshape(-1,3)
            grid = grid[np.lexsort((np.abs(grid[:,2]),
                                    np.abs(grid[:,1]),
                                    np.abs(grid[:,0])))]
            return grid[1:]


        # First search
        a_mag = np.linalg.norm(np.dot([maxindex, maxindex, maxindex], structure.lattice))
        c_angle = 90
        a_uvw = None
        c_uvw = None
        for uvw in gen_grid(maxindex):
            cart = np.dot(uvw, structure.lattice)
            mag = np.linalg.norm(cart)
            angle = 180 * vec_angle(cart, planenormal) / np.pi

            # Find shortest vector in the plane
            if np.isclose(np.dot(cart, planenormal), 0.0):
                if mag < a_mag:
                    a_uvw = uvw
                    a_mag = mag

            # Find vector closest to plane normal 
            elif angle < c_angle:
                c_angle = angle
                c_uvw = uvw

        assert a_uvw is not None, 'Failed to find first vector in slip plane'
        assert c_uvw is not None, 'Failed to find vector near slip plane normal'

        # Reduce c_uvw if possible
        c_uvw = c_uvw / np.gcd.reduce(np.asarray(c_uvw, dtype=int)) 

        # Second search
        a_cart = np.dot(a_uvw, structure.lattice)
        b_mag = np.linalg.norm(np.dot([maxindex, maxindex, maxindex], structure.lattice))
        b_uvw = None
        min_angle = 180.0
        for uvw in gen_grid(maxindex):
            cart = np.dot(uvw, structure.lattice)
            angle = 180 * vec_angle(a_cart, cart) / np.pi
            # Check that vector is in plane and not parallel to a_uvw
            if np.isclose(np.dot(cart, planenormal), 0.0) and not np.isclose(angle, 0.0) and not np.isclose(angle, 180.0):

                # Check if right-handed
                if np.dot(np.cross(a_cart, cart), planenormal) > 0:

                    # Find b_uvw with smallest magnitude and smallest angle
                    mag = np.linalg.norm(cart)
                    if (np.isclose(mag, b_mag) and angle < min_angle) or mag < b_mag:
                        b_uvw = uvw
                        b_mag = mag
                        min_angle = angle

        assert b_uvw is not None, 'Failed to find second vector in slip plane'

        return a_uvw, b_uvw, c_uvw
    
    def surface(self, hkl, maxindex=None, orth=False, saxis='c', **kwargs):
        """
        Arguments:
            hkl: The free surface plane to generate expressed in 3 indices
                 Miller (hkl) format
        """
        hkl = np.array(hkl)
        structure=self._structure
        # check 
        if hkl.shape != (3,):
            raise ValueError('Miller indices must be 3 values')
        if np.allclose(hkl, np.asarray(hkl, dtype=int)):
            hkl = np.asarray(hkl, dtype=int)
        else:
            raise ValueError('Miller indices must be integers')
        a_uvw, b_uvw, c_uvw = self._surface(structure, hkl, maxindex)
        if orth:
            bound = maxindex if maxindex else 10
            print(a_uvw @ structure.lattice, b_uvw @structure.lattice)
            det, coeffs = self.find_minimal_orthogonal_lattice_vectors(a_uvw@structure.lattice, b_uvw@structure.lattice, bound=bound)
            if det == None:
                raise RuntimeError("Find orthogonal_lattice_vectors failed.")
            print(det, coeffs)
            #a_uvw = a_uvw * coeffs[0] + b_uvw * coeffs[1]
            #b_uvw = a_uvw * coeffs[2] + b_uvw * coeffs[3]
            a_uvw, b_uvw = a_uvw * coeffs[0] + b_uvw * coeffs[1], a_uvw * coeffs[2] + b_uvw * coeffs[3]

        # Orient the uvw sets based on saxis 
        if saxis == 'c':
            uvws = np.array([a_uvw, b_uvw, c_uvw], dtype=int)
        elif saxis == 'b':
            uvws = np.array([b_uvw, c_uvw, a_uvw], dtype=int)
        elif saxis == 'a':
            uvws = np.array([c_uvw, a_uvw, b_uvw], dtype=int)

        # Verify the right-hand rule
        new_lattice = uvws @ structure.lattice
        if np.linalg.det(new_lattice) < 0:
            uvws[1] = uvws[1] * -1   

        return self.redefine(uvws)

    def twin_crystal(self, axis='c', spacing=0, coherent=False, tol=1, **kwargs):
        """
        """
        lattice = self.structure.lattice
        positions = self.structure.get_positions(type='direct')
        positions -= np.floor(positions)
        elements = self.structure.get_elements(type="symbol")
        lattice_parameters = self.structure.lattice_parameters
        natom = len(self.structure)

        for i,ax in enumerate('abc'):
            if ax == axis:
                shift = np.array([0,0,0])
                shift[i] = 1
                mirror = 1-2*shift
                lp = lattice_parameters[i]
                spshift = shift*spacing/(lp+spacing*2)
                new_positions = np.r_[positions+spshift/2, positions*mirror+shift*2 - spshift/2]
                new_positions[:,i] /= 2
                new_lattice = np.diag(shift+2*spshift+1) @ lattice
                ax = i
                break
        else:
            raise ValueError("Invalid axis setting. Use a/b/c please")

        # merge atoms
        coords = new_positions[:,ax].reshape(2,-1)
        d = coords[1]-coords[0]
        d = abs(d - np.around(d)) * lp
        indices1 = np.where(d > tol)[0]
        ts = elements[np.where(d < tol)[0]]
        #print(coords[:,0], 1-coords[:,0])
        #assert abs(coords[0,0]) == abs(1-coords[1,0])

        if coherent:
            indices = np.r_[np.arange(natom), indices1+natom]
            indices2 = np.where(d-spacing <= tol)[0]
        else:
            indices = np.r_[indices1, indices1+natom]
            indices2 = np.where(d <= tol)[0]

        #print(len(positions))
        #print(len(indices))
        #print(len(indices2))

        if coherent:
            add_positions = new_positions[indices2] 
            add_positions[:,ax] = add_positions[:,ax] + 0.5 - spacing/(spacing+lp)/2 
            add_elements = elements[indices2]
            new_positions = new_positions[indices]
            new_elements = np.tile(elements, 2)[indices]
            new_positions = np.r_[new_positions, add_positions]
            new_elements = np.r_[new_elements, add_elements]
        else:
            add_positions1 = np.array(new_positions[indices2])
            add_positions2 = np.array(new_positions[indices2])
            add_positions1[:,ax] = 0.5
            add_positions2[:,ax] = 0.0
            add_elements = elements[indices2]
            #print(np.unique(add_elements, return_counts=True))
            new_positions = new_positions[indices]
            new_elements = np.tile(elements, 2)[indices]
            #print(np.unique(new_elements, return_counts=True))
            new_positions = np.r_[new_positions, add_positions1, add_positions2]
            new_elements = np.r_[new_elements, add_elements, add_elements]

        self._structure = Structure.from_cell((new_lattice, new_positions, new_elements))
        return self
    
    def adsorption(self):
        """
        """
        pass

    def images(self, structure, numbers_of_images, **kwargs):
        """        
     
        """
        from ..materials.structure import Structure
        
        structure=self._structure
        
        # check
        if not isinstance(structure, Structure): 
            raise ValueError('unrecognized structure')
        
        images=[] # image structures
        
        return self

    def rotation(self, atoms, axis, theta, **kwargs):
        """
        rotation given atoms.
        
        arguments:
            axis: rotation axis. Note that, for molecule, the type of axis is only 'Cartesian'. The valid format:
                [0.1, 0.0, 0.0, 'Direct']
                [0.1, 0.0, 0.0]
                [5.234, 0.0, 0.0, 'Cartesian']
            theta: rotation angle. The valid format:
                [30, 'Degree']
                [0.2, 'Radian']
            atoms: collection of atom's formated atom or object. i.e. [atom0, atom1, atom2,...] 
                    ['Na', 0.1, 0.0, 0.0, 'Direct']
                    ['Na', 0.1, 0.0, 0.0]
                    ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                    
                    contain species information:
                    ['Na1+', 0.1, 0.0, 0.0, 'Direct']
                    ['Na1+', 0.1, 0.0, 0.0]
                    ['Na1+', 5.234, 0.0, 0.0, 'Cartesian']
            isUpdatedInfo (default=False): whether to update the composition and symmetry information (include the site, operation, wyckoffSite, spacegroup).
            
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
                origin: rotation origin. Noth that it is the origin of the axis of rotation, not a point on the axis of rotation.
                    The valid format:
                    [0.1, 0.0, 0.0, 'Direct']
                    [0.1, 0.0, 0.0]
                    [5.234, 0.0, 0.0, 'Cartesian']
                    
        Returns:
            structureFactory's object.
        """
        structure=self._structure
        
        # formatd axis (direct)
        axis = formated_fraction_vector(axis, structure.lattice)

        # formated origin (direct)
        if 'origin' in kwargs: 
            if isinstance(structure, Structure):
                origin=formated_fraction_vector(kwargs['origin'], structure.lattice) @ structure.lattice
            else:
                origin=formated_cartesian_vector(kwargs['origin'], structure.lattice)
        else:
            origin=np.zeros(3,)

        rotation_matrix = formated_rotation_matrix(theta, axis)

        # get atom indices % 
        indices = []
        for atom in atoms:
            if isinstance(atom, (int, np.int32, np.int64)):
                indices.append(atom)
            elif isinstance(atom, Atom):
                index = structure.get_atom(atom.scale_coord, specie=atom.specie)
                indices.extend(index)
            else:
                index = structure.get_atom(atom)
                indices.extend(index)

        # atom_positions %
        positions = np.array([structure.atomic_positions[i].coord for i in indices])
        new_positions = (positions - origin) @ rotation_matrix + origin

        for i,j in enumerate(indices):
            structure.atomic_positions[j].coord = new_positions[i]
 
        #structure.update()
        self._structure=structure
        return self
    
    def translation(self, atoms=None, direction=None, **kwargs):
        """
        translation given atoms.
        
        arguments:
            direction: direction vector to add the vacuum along lattice vector(a/b/c). The valid format is :
                [0.1, 0, 0, 'Direct']
                [0.1, 0, 0] (for Direct, can not be specify)
                [5.234, 0, 0, 'Cartesian'] (for Cartesian, must be given)
            atoms: collection of atom's formated atom or object. i.e. [atom0, atom1, atom2,...] 
                    ['Na', 0.1, 0.0, 0.0, 'Direct']
                    ['Na', 0.1, 0.0, 0.0]
                    ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                    
                    contain species information:
                    ['Na1+', 0.1, 0.0, 0.0, 'Direct']
                    ['Na1+', 0.1, 0.0, 0.0]
                    ['Na1+', 5.234, 0.0, 0.0, 'Cartesian']
            isUpdatedInfo (default=False): whether to update the composition and symmetry information (include the site, operation, wyckoffSite, spacegroup).
            
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
        
        Returns:
            structureFactory's object.
        """        
        structure=self._structure
        
        # formatd axis (direct)
        if direction is None:
            raise ValueError("Need directions.")
        axis = formated_fraction_vector(direction, structure.lattice)
        
        # get atom indices % 
        if atoms is None:
            indices = np.arange(len(structure))
        else:
            indices = []
            for atom in atoms:
                if isinstance(atom, (int, np.int32, np.int64)):
                    indices.append(atom)
                elif isinstance(atom, Atom):
                    index = structure.get_atom(atom.scale_coord, specie=atom.specie)
                    indices.extend(index)
                else:
                    index = structure.get_atom(atom)
                    indices.extend(index)
       
        # atom_positions %
        positions = np.array([structure.atomic_positions[i].scale_coord for i in indices])
        new_positions = positions + direction

        for i,j in enumerate(indices):
            structure.atomic_positions[j].scale_coord = new_positions[i]
 
        #structure.update()
        self._structure=structure
        return self
    
    # for Molecular dynamics
    def initializeVelocityDistribution(self, temperature, **kwargs):
        """
        initialize the velocity (unit: angstrom/fs) distribution of atoms at given temperature.
        
        arguments:
            temperature: desired temperature (unit: K).
            
        Returns:
            structureFactory's object.
        """
        structure=self._structure
        
        momentum=[0,0,0] # sum of velocities
        ke=0 # kinetic energy 
        
        natoms = len(structure)
        velocities = np.random.random((natoms,3)) # Angstrom/fs
        momentum = np.mean(velocities, axis=0)
        velocities -= momentum
        for i,atom in enumerate(structure.atomic_positions):
            ke += atom.elementinfo.mass*np.sum(velocities[i]**2)*1e7 # Kg x (m/s)^2
        R=8.3144598 # Gas constant (J/Kmol)
        t0=ke/(R*3*(natoms-1)) # calculated temperature
        scale=np.sqrt(temperature/t0)
        
        # scale to the desired temperature
        for atom in structure.atomic_positions:
            atom.velocity=scale*(atom.velocity-momentum)
            
        return self
    
    def perturb(self, cutoff=0.1, **kwargs):
        """
        perturb the atomic position.
        
        arguments:
            cutoff (default=0.1): cutoff of perturbation (unit: Angstrom).  
        """
        
        structure=self._structure
        
        natoms=len(structure)
        perturbations=np.random.randn(natoms,3)*cutoff # unit: Angstrom
        
        # perturb positions
        for i,atom in enumerate(structure.atomic_positions):
            position = atom.coord + perturbations[i]
            atom.coord = position
        
        return self
    
    def getUnit(self, unit, tolerance=0.1, **kwargs):
        """
        get a unit with giving range.
        
        Arguments:
            unit: unit of operation. The valid format is [start, end, direction(0/1/2)]. i.e. [0.24980, 0.31235, 2] or [[0.1, 0.2],
                                                                                                                        [0.1, 0.2],
                                                                                                                        [0.1, 0.2]]
            tolerance (default=0.1): tolerance for unit to exclude the right atoms near boundary (unit: Angstrom).
            
            kwargs:
                symbol_of_atoms:symbol of atoms. i.e. ['Na', 'Cl']
        """
        from ..utils.check import is_inside

        structure=self._structure
        
        symbol_of_atoms=None
        if 'symbol_of_atoms' in kwargs:
            symbol_of_atoms=kwargs['symbol_of_atoms']
        
        # delete unit
        unit_atoms=[]
        for atom in list(structure.atoms):
            position=np.array(atom.position)
            
            if is_inside(position=position, selective_range=unit): # be careful on the left boundary
                formated_atom0=atom.to_formated_atom()
                if (symbol_of_atoms is None) or (formated_atom0[0] in symbol_of_atoms):
                    unit_atoms.append(formated_atom0)
        return unit_atoms
        
    
    def removeUnit(self, unit, tolerance=0.1, isMoveAtoms=True, **kwargs):
        """
        remove a unit with giving range.
        
        Arguments:
            unit: unit of operation. The valid format is [start, end, direction(0/1/2)]. i.e. [0.24980, 0.31235, 2] (Direct-type for position)
            tolerance (default=0.1): tolerance for unit to exclude the right atoms near boundary (unit: Angstrom).
            isMoveAtoms (default=True): whether to move right atoms to fill the space caused by cutting unit.
            
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
                
                for isMoveAtoms (optional)
                    isPersistLattice (defalt=True): whether persist the lattice parameters when cutting unit. if False, the lattice will shrink by the length of cutting unit.
        """
        from ..utils.convert import direct2cartesian
        structure=self._structure
        
        # vector of left unit
        ul=np.array([0.0, 0.0, 0.0])
        ul[unit[2]]=unit[0]
        # vector of left unit
        ur=np.array([0.0, 0.0, 0.0])
        ur[unit[2]]=unit[1]
        
        # delete unit
        unit_atoms=[]
        for atom in list(structure.atoms):
            position=np.array(atom.position)
            d0=direct2cartesian(structure.lattice, position-ul) # distance from left boundary
            d1=direct2cartesian(structure.lattice, ur-position) # distance from right boundary
            if d0[unit[2]] >= -tolerance and d1[unit[2]] > tolerance: # be careful on the left boundary
                unit_atoms.append(atom.to_formated_atom())
        self.del_atoms(unit_atoms)
        
        # move atoms
        if isMoveAtoms:
            moving_atoms=[]
            for atom in list(structure.atoms):
                d1=direct2cartesian(structure.lattice, ur-atom.position) # distance from right boundary
                if d1[unit[2]] <= tolerance: moving_atoms.append(atom.to_formated_atom())
            direction=[0.0,0.0,0.0]
            direction[unit[2]]=-(unit[1]-unit[0])
            self.translation(atoms=moving_atoms, direction=direction, isCheckOverlap=False)
            
            isPersistLattice=True
            if 'isPersistLattice' in kwargs: isPersistLattice=kwargs['isPersistLattice']
            # shrink lattice along given direction
            if not isPersistLattice: self.vacuum(direction=direction, isCenter=False)
        
        self._structure=structure
        return self
    
    def addUnit(self, unit, nrepeat, tolerance=0.1, **kwargs):
        """
        add repeat unit with giving range along a direction.
        
        Arguments:
            unit: unit of operation. The valid format is [start, end, direction(0/1/2)]. i.e. [0.24980, 0.31235, 2] (Direct-type for position)
            nrepeat: number of repeat.
            tolerance (default=0.1): tolerance for unit to exclude the right atoms near boundary (unit: Angstrom).
            
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
                
                isPersistLattice (defalt=False): whether persist the lattice parameters when cutting unit.
        """
        from copy import deepcopy
        from ..utils.convert import direct2cartesian
        structure=self._structure
        
        # add vacuum with a length of nrepeat
        direction=[0.0, 0.0, 0.0]
        direction[unit[2]]=nrepeat*(unit[1]-unit[0])
        
        # prolong lattice along given direction
        isPersistLattice=False
        if 'isPersistLattice' in kwargs: isPersistLattice=kwargs['isPersistLattice']
        if isPersistLattice:
            atoms=sorted(structure.atoms, key=lambda atom: atom.position[unit[2]])
            distance=1.0-atoms[-1].position[unit[2]] # Direct
            if distance < direction[unit[2]]: raise ValueError('not engouth space to insert the untis along given direction.\nhaving: {:.4f}; needing: {:4f}'.format(distance, direction[unit[2]]))
        else:
            self.vacuum(direction=direction, isCenter=False)
            unit[0] /= (direction[unit[2]]+1.0)
            unit[1] /= (direction[unit[2]]+1.0)
            direction[unit[2]] /= (direction[unit[2]]+1.0)
            
        # vector of left unit
        ul=np.array([0.0,0.0,0.0])
        ul[unit[2]]=unit[0]
        # vector of left unit
        ur=np.array([0.0,0.0,0.0])
        ur[unit[2]]=unit[1]
        
        # move atoms
        moving_atoms=[]
        for atom in list(structure.atoms):
            d1=direct2cartesian(structure.lattice, ur-atom.position)
            if d1[unit[2]] <= tolerance: moving_atoms.append(atom.to_formated_atom())
        self.translation(atoms=moving_atoms, direction=direction, isCheckOverlap=False)
        
        # add unit
        unit_atoms=[]
        for atom in list(structure.atoms):
            d0=direct2cartesian(structure.lattice, atom.position-ul) # distance from left boundary
            d1=direct2cartesian(structure.lattice, ur-atom.position) # distance from right boundary
 
            if d0[unit[2]] >= -tolerance and d1[unit[2]] > tolerance: # be careful on the left boundary
                unit_atoms.append(atom.to_formated_atom())

        add_atoms=[]        
        for i in range(1, nrepeat+1):
            for atom in unit_atoms:
                tmp=deepcopy(atom)
                tmp[unit[2]+1] += i*(unit[1]-unit[0])
                add_atoms.append(tmp)
        self.add_atoms(atoms=add_atoms)

        self._structure=structure
        return self
    
    def insertMolecule(self, structure_of_molecule, position_in_molecule, position_in_structure, **kwargs):
        """
        insert a molecule to structure.
        
        Arguments:
            structure_of_molecule: molecule's object.
            position_in_molecule: reference position in molecule for moving. the valid format is : [5.234, 0, 0, 'Cartesian']
            position:reference position in structure for moving. the valid format is :
                [0.1, 0, 0, 'Direct']
                [0.1, 0, 0] (for Direct, can not be specify)
                [5.234, 0, 0, 'Cartesian'] (for Cartesian, must be given)
            isUpdatedInfo (default=False): whether to update the composition and symmetry information (include the site, operation, wyckoffSite, spacegroup).
            
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
        
        Returns:
            structureFactory's object.
        """
        structure=self._structure
        molecule=structure_of_molecule
        
        # molecule position
        if is_list_or_array(position_in_molecule):
            position_in_molecule = formated_cartesian_vector(position_in_molecule, molecule.lattice)

        elif isinstance(position_in_molecule, Atom):
            position_in_molecule = position_in_molecule.coord
           
        if is_list_or_array(position_in_structure):
            position_in_structure = formated_fraction_vector(position_in_structure, structure.lattice)

        elif isinstance(position_in_structure, Atom):
            position_in_structure = position_in_structure.scale_coord

        direction = position_in_structure @structure.lattice - position_in_molecule
        
        positions = molecule.get_positions(type='cartesian') + direction
        species = molecule.get_elements(type='symbol')
        self.add_atoms(positions=positions, species=species, position_format='cartesian')
        
        structure.update()
        return self
    
    
def normalized(vectors):
    vectors = vectors - np.floor(vectors)
    vectors = np.around(vectors,8)
    vectors = vectors - np.around(vectors)
    return vectors

def remove_duplicates_with_tolerance(vectors, tolerance):
    from scipy.spatial import cKDTree
    vectors = np.array(vectors)
    
    tree = cKDTree(vectors)
    duplicates = np.zeros(len(vectors), dtype=bool)
    
    for i, vec in enumerate(vectors):

        if duplicates[i]:
            continue
    #    print(i, vec)

        neighbors = tree.query_ball_point(vec, tolerance)
        if i in neighbors:
            neighbors.remove(i)
        
        #print(i, neighbors)
        if neighbors:
            duplicates[neighbors] = True
            
    return duplicates
    #unique_vectors = vectors[~duplicates]
    #return unique_vectors

# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
from ..utils.variables import default_constants


class StructureFactory(object):
    
    def __init__(self, structure, isOperateOnSelf=False, isPersist=False, **kwargs):
        """
        Arguments:
            structure: structure's object.
            isOperateOnSelf: Whether to operate itself.
            isPersist (default=False): whether to save to the database.
            
            kwargs:
                isCloneFullInfo (default=False): whether to clone all information of structure.
        """
        self._raw_structure=structure
        
        self._structure=None
        if isOperateOnSelf:
            self.structure=structure
        else:
            isCloneFullInfo=kwargs.get('isCloneFullInfo',False)
            self._structure=structure.minimize_clone() if isCloneFullInfo else structure.clone()
            
        if isPersist: self._structure.update(isPersist=isPersist)
        
    @property
    def raw_structure(self):
        """
        raw structure.
        """
        from ..utils.check import isOverlap
        
        if not isOverlap(structure=self._raw_structure):
            return self._raw_structure
        else:
            raise ValueError('exist position overlap in structure.atoms')
        
    @property
    def structure(self):
        """
        structure object after operation.
        """
        from ..utils.check import isOverlap
        
        if not isOverlap(structure=self._structure):
            return self._structure
        else:
            raise ValueError('exist position overlap in structure.atoms')
        
    
    def zoom(self, scale, isPersist=False):
        """
        scale the lattice vector.
        
        Arguments:
            scale: coefficient of zoom for lattice parameters.
            isPersist (default=False): whether to save to the database.
            
        Returns:
            structureFactory's object.
        """
        structure=self._structure
        lattice=structure.lattice
        structure.lattice=lattice*scale

        structure.update(isPersist=isPersist)
        self._structure=structure
        return self
    
    def scale(self, direction, isPersist=False):
        """
        scale the lattice vector by given direction.
        
        Arguments:
            direction: coefficient of zoom for lattice parameters along three axes. i.e. [0.9, 1, 1], [0.9, 0.9, 1], [0.9, 0.9, 0.9]
            isPersist (default=False): whether to save to the database.
            
        Returns:
            structureFactory's object.
        """
        structure=self._structure
        
        # check
        if False in [True if d0 > 0 else False for d0 in direction]: raise ValueError('unrecognized direction')
        
        lattice=structure.lattice
        for i in range(0, 3):
            structure.lattice[i]=lattice[i]*direction[i]

        structure.update(isPersist=isPersist)
        self._structure=structure
        return self
    
    def add_atoms(self, atoms, isUpdatedInfo=False, isPersist=False, **kwargs):
        """
        add atoms to structure.
        
        Arguments:
            atoms: collection of atom's object or formated string. i.e. [atom0, atom1, atom2,...] 
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
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
        
        Returns:
            structureFactory's object.
        """
        from ..utils.check import check_formated_atom
        from ..materials.atom import Atom
        
        # remove atomic translation periodicity
        isNormalizingCoordinate=kwargs.get('isNormalizingCoordinate',default_constants.isNormalizingCoordinate.value)
        precision=kwargs.get('precision',default_constants.precision.value)
        
        structure=self._structure
        for atom in atoms:
            formated_atom=None
            if isinstance(atom, Atom):
                formated_atom=atom.to_formated_atom()
            elif check_formated_atom(atom):
                formated_atom=atom
            else:
                raise ValueError('unrecognized atom')
            structure.add_atom(Atom().create(formated_atom=formated_atom, lattice=structure.lattice), isNormalizingCoordinate=isNormalizingCoordinate, precision=precision)
        
        # default
        symprec=kwargs.get('symprec',default_constants.symprec.value)
        angle_tolerance=kwargs.get('angle_tolerance',default_constants.angle_tolerance.value)
        
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    def del_atoms(self, atoms, isUpdatedInfo=False, isPersist=False, **kwargs):
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
        from ..utils.check import check_formated_atom
        from ..materials.atom import Atom
        
        structure=self._structure
        for atom in list(atoms):
            formated_atom=None
            if isinstance(atom, Atom):
                formated_atom=atom.to_formated_atom()
            elif check_formated_atom(atom):
                formated_atom=atom
            else:
                raise ValueError('unrecognized atom')
            
            atom0=structure.get_atom(formated_atom)
            if atom0 is None:
                raise ValueError('not exist in structure.atoms: {}'.format(formated_atom))
            else:
                structure.del_atom(atom0)

        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    def substitute_atoms(self, atoms, symbol_of_elements, isUpdatedInfo=False, isPersist=False, **kwargs):
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
        from ..utils.check import check_formated_atom
        from ..materials.atom import Atom
        
        structure=self._structure
        for i in range(0, len(atoms)):
            formated_atom=None
            atom=atoms[i]
            if isinstance(atom, Atom):
                formated_atom=atom.to_formated_atom()
            elif check_formated_atom(atom):
                formated_atom=atom
            else:
                raise ValueError('unrecognized atom')
            atom=structure.get_atom(formated_atom=formated_atom)
            if atom == None:
                raise ValueError('no match atom %s' %formated_atom)
            
            if isinstance(symbol_of_elements, str):
                symbol_of_element=symbol_of_elements
                structure.substitute_atom(atom=atom, symbol_of_element_or_species=symbol_of_element)
            elif isinstance(symbol_of_elements, (list, np.ndarray)):
                symbol_of_element=symbol_of_elements[i]
                structure.substitute_atom(atom=atom, symbol_of_element_or_species=symbol_of_element)
        
        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    def center(self, direction, dtype_of_move='position', isUpdatedInfo=False, isPersist=False, **kwargs):
        """
        move atoms to center by given direction.
        
        arguments:
            direction: direction vector. The valid format is [1, 0, 0], [0, 1, 0], [0, 0, 1].
            type_of_move (default='position'): type of move. 
                'mass': moving by center of mass.
                'position': moving by boundary of position.
            isUpdatedInfo (default=False): whether to update the composition and symmetry information (include the site, operation, wyckoffSite, spacegroup).
            isPersist (default=False): whether to save to the database.
            
        Returns:
            structureFactory's object.
        """
        from ..utils.check import check_formated_position
        
        structure=self._structure
        
        # check
        if not(check_formated_position(direction) and (direction.count(1) == 1) and (direction.count(0) == 2)): raise ValueError('unknown direction') #[1, 0, 0], [0, 1, 0], [0, 0, 1]
        
        dindex=direction.index(1)
        atoms=structure.atoms
        if dtype_of_move.lower().startswith('m'): # mass
            sum4mr=0.0
            total_mass=0.0
            for atom0 in atoms:
                sum4mr += atom0.element.mass*atom0.position[dindex] # sum of mi x ri
                total_mass += atom0.element.mass
            mass_center=sum4mr/total_mass
# =============================================================================
#             # Note that must updte the structural information to ensure that all information is synchronized.
#             structure.update()
#             mass_center=sum4mr/structure.mass
# =============================================================================
            for atom0 in atoms:
                atom0.position[dindex] -= (mass_center-0.5)
        elif dtype_of_move.lower().startswith('p'): # position
            pos=[atom0.position[dindex] for atom0 in atoms]
            for atom0 in atoms:
                atom0.position[dindex] -= ((np.max(pos)+np.min(pos))/2-0.5)
        else:
            raise ValueError('unknown type')

        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    def vacuum(self, direction, isUpdatedInfo=False, isPersist=False, **kwargs):
        """
        add vacuum along a direction.
        
        arguments:
            direction: direction vector to add the vacuum along lattice vector(a/b/c). The valid format is :
                [0.1, 0, 0, 'Direct']
                [0.1, 0, 0] (for Direct, can not be specify)
                [5.234, 0, 0, 'Cartesian'] (for Cartesian, must be given)
            isUpdatedInfo (default=False): whether to update the composition and symmetry information (include the site, operation, wyckoffSite, spacegroup).
            isPersist (default=False): whether to save to the database.
            
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
        from ..utils.check import check_formated_position
        from ..utils.convert import any2cartesian
        structure=self._structure
        
        # check
        if not check_formated_position(direction): raise ValueError('unknown direction')
        
        direction=any2cartesian(structure.lattice, direction)[:-1]
        mod=np.sum(direction) # summation of direction
        msod=np.sum(np.abs(direction)) # summation of absolute direction
        if mod != msod or msod == 0:
            if -mod == msod:
                import warnings
                warnings.warn('Warning: compressing the vacuum')
            else:
                raise ValueError('unknown direction')
        
        isCenter=kwargs.get('isCenter',True)
        distance=kwargs.get('distance',None)
            
        lattice=structure.lattice
        lattice_parameters=structure.lattice_parameters
        # add vacuum layer
        for i in range(0, len(direction)):
            if direction[i] != 0: # vacuum direction
                scale=direction[i]/lattice_parameters[i] # part of vacuum
                lattice[i]=lattice[i]*(1+scale)
                for atom in structure.atoms:
                    atom.position[i] /= (1+scale)

        # move atoms
        for i in range(0, len(direction)):
            if direction[i] != 0: # vacuum direction
                prange=[] # range of position along given direction.
                for atom in structure.atoms:
                    prange.append(atom.position[i])
                    
                scale=None
                if distance != None:
                    scale=distance/structure.lattice_parameters[i]
                    if scale+np.max(prange) > 1.0: raise ValueError('distance is too large and move to the boundary')
                elif isCenter:
                    center_of_atoms=(np.max(prange)+np.min(prange))/2
                    scale=0.5-center_of_atoms
                    print('center', center_of_atoms)
                if scale != None:
                    for atom in structure.atoms:
                        atom.position[i] += scale
        
        structure.lattice=lattice
        structure.volume=structure.calculate_volume()
        structure.volume_per_atom=structure.volume/structure.natoms
        
        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    def magnetism_order(self, element_magmoms, isPersist=False):
        """
        At present, only consider FM configuration. Other magnetic configuration need to set the atomic magnetism by hand.
        
        Arguments
            element_magmoms: dictionary of element's symbol and its magnetic moment. The valid formation:
                {'Fe':5,
                 'Cr':3,
                ...}
            isPersist (default=False): whether to save to the database.
            
        Returns:
            structureFactory's object.
        """
        structure=self._structure
        for element in structure.elements:
            for atom in structure.get_atoms_of_element(symbol=element.symbol):
                if element.symbol in element_magmoms:
                    atom.magmom=element_magmoms[element.symbol]
                else:
                    atom.magmom=0.0

        structure.update(isPersist=isPersist)
        self._structure=structure
        return self
    
    def constraint(self, atoms, isPersist=False, isConstraint_of_remainder=True, **kwargs):
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
    
            isPersist (default=False): whether to save to the database.
            isConstraint_of_remainder (default=False): whether to update constraint of remainder atoms
                
            kwarges:
                constraints: constraints of given atoms.
                constraint_of_remainder (default=[True, True, True]): constraint of remainder atoms.
            
        Returns:
            structureFactory's object.
        """
        from ..utils.check import check_constraint, check_formated_atom
        from ..materials.atom import Atom
        
        # remove atomic translation periodicity
#        isNormalizingCoordinate=True
        isNormalizingCoordinate=default_constants.isNormalizingCoordinate.value
        if 'isNormalizingCoordinate' in kwargs: isNormalizingCoordinate=kwargs['isNormalizingCoordinate']
        precision=default_constants.precision.value
        if 'precision' in kwargs: precision=kwargs['precision']
        #constraints=kwargs['constraints'] if 'constraints' in kwargs else None
        constraints=None
        if 'constraints' in kwargs:
            constraints = np.array(kwargs['constraints'])
            if not ( constraints.shape == (3,) or constraints.shape == (len(atoms),3) ):
                raise ValueError('unknown constraints')
        
        structure=self._structure

        atoms0=[]
        constraints0=[]
        for i in range(0, len(atoms)):
            formated_atom=atoms[i]
            if constraints is None:
                constraint=formated_atom[-3:]
                formated_atom=formated_atom[:-3]
            else:
                if len(constraints.shape) == 1: # constraints: [True, True, True]
                    constraint=constraints        
                elif len(constraints.shape) == 2: # constraints: [[True, True, True], [True, True, True], ...]
                    constraint=constraints[i]
                if formated_atom[4].lower().startswith('c') or formated_atom[4].lower().startswith('d'): # ['Na1+', 0.1, 0.0, 0.0, 'Direct', True, True, False]
                    formated_atom=formated_atom[:4]
                else: #  ['Na1+', 0.1, 0.0, 0.0, True, True, False]
                    formated_atom=formated_atom[:3]
                    
            atom=None
            if isinstance(formated_atom[0], Atom):
                atom=structure.get_atom(formated_atom=formated_atom[0].to_formated_atom(), 
                                        isNormalizingCoordinate=isNormalizingCoordinate, 
                                        precision=precision)
            elif check_formated_atom(formated_atom):
                atom=structure.get_atom(formated_atom=formated_atom, 
                                        isNormalizingCoordinate=isNormalizingCoordinate, 
                                        precision=precision)
                if atom is None: raise ValueError('non-exist atom in structure.atoms: {}'.format(formated_atom))
            else:
                raise ValueError('unknown atom in atoms: {}'.format(atom))
            
            if not check_constraint(constraint): raise ValueError('unknown constrain of atom in atoms')
            
            # atoms with given constrain information
            atoms0.append(atom)
            constraints0.append(constraint)

        # update constrain information        
        constraint_of_remainder=[True, True, True]
        if 'constraint_of_remainder' in kwargs:
            constraint_of_remainder=kwargs['constraint_of_remainder']
            if not check_constraint(constraint_of_remainder): raise ValueError('unknown constrain of atom in atoms')
        for atom in structure.atoms:
            if atom in atoms0:
                atom.constraint=constraints0[atoms0.index(atom)]
            elif isConstraint_of_remainder is True:
                atom.constraint=constraint_of_remainder
            
        structure.update(isPersist=isPersist)
        self._structure=structure
        return self
    
    def redefine(self, operator_matrix, isPersist=False):
        """
        redefine lattcie cell: C'=C x M.
        
        Arguments:
            operator_matrix: operator matrix (M). The valid formation:
                [[0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 0]]
                Note that the component of M should be integer. And the volume of M is an integer greater than 0.
            isPersist (default=False): whether to save to the database. 
                   
        Returns:
            structureFactory's object.
        """
        import numbers
        from ..utils.check import is_inside
        from ..utils.convert import any2cartesian, any2direct, normalize_position, cell2poscar
        from ..materials.structure import Structure
        
        structure=self._structure
        poscar = self._structure.formatting('cell')
        
        operator_matrix=np.array(operator_matrix)
        # check
        if operator_matrix.shape != (3,3): raise ValueError('invalid operator_matrix')
        for i in range(0, operator_matrix.shape[0]):
            for j in range(0, operator_matrix.shape[1]):
                if not isinstance(operator_matrix[i][j], numbers.Integral): raise ValueError('contain non-integer in operator_matrix')
        if np.abs(np.linalg.det(operator_matrix)) < 1e-4: raise ValueError('calculated volume by operator_matrix must be larger than 0')
        
        lattice = poscar['lattice']
        lattice_new = operator_matrix.T @ lattice
# =========================================================================
#        lattice_new=[]
#        for i in range(0, 3):
#            lattice_new.append(lattice[0]*operator_matrix[i][0]+lattice[1]*operator_matrix[i][1]+lattice[2]*operator_matrix[i][2])
#==========================================================================      

        '''
        elements = poscar['numbers']
        positions = poscar['positions']

        def get_range(array):
            dmin = sum(np.where(array<0, array, 0))
            dmax = sum(np.where(array>0, array, 0))
            return int(np.ceil(max(-dmin, dmax)))

        def normalized(vectors):
            vectors = vectors - np.floor(vectors)
            vectors = np.around(vectors,8)
            vectors = vectors - np.around(vectors)
            return vectors
        
        # suerpcell
        dim1 = get_range(operator_matrix[:,0])
        dim2 = get_range(operator_matrix[:,1])
        dim3 = get_range(operator_matrix[:,2])
        grid = np.mgrid[-dim1:dim1+1, -dim2:dim2+1, -dim3:dim3+1].reshape(3,-1).T # x,3
        grid_positions = positions[None,:,:] + grid[:,None,:] # ngrid, natom, 3
        cart_positions = normalized(grid_positions @ lattice @ np.linalg.inv(lattice_new))
 
        new_elements = []
        new_positions = []
 
        for i in range(cart_positions.shape[0]):
            for j in range(cart_positions.shape[1]):
                coord = cart_positions[i,j]
                for position in new_positions:
                    if np.sum(np.abs(normalized(position-coord))) < 1e-3:
                        break
                else:
                    new_elements.append(elements[j])
                    new_positions.append(coord)
                    
        positions=np.array(new_positions)
        numbers=np.array(new_elements)
        '''
        positions=[]
        numbers=[]
        # suerpcell
# =========================================================================
#         dim=operator_matrix[0]+operator_matrix[1]+operator_matrix[2]
# =========================================================================
        operator_matrix2=np.vstack((operator_matrix,
                                    operator_matrix[0]+operator_matrix[1], # a+b
                                    operator_matrix[0]+operator_matrix[2], # a+c
                                    operator_matrix[1]+operator_matrix[2], # b+c
                                    operator_matrix[0]+operator_matrix[1]+operator_matrix[2]))

        #dim=[[np.min(operator_matrix2[:, ii]), np.max(operator_matrix2[:, ii])] for ii in range(0, 3)]
        dim=[[-np.max(np.abs(operator_matrix2[:, ii])), np.max(np.abs(operator_matrix2[:, ii]))] for ii in range(0, 3)]
        for atom in structure.atoms:
            position=atom.position
            for i in range(dim[0][0], dim[0][1]): # x
                for j in range(dim[1][0], dim[1][1]): # y
                    for k in range(dim[2][0], dim[2][1]): # z
                        x=position[0]+i
                        y=position[1]+j
                        z=position[2]+k
                        p=any2cartesian(lattice, [x, y, z])#.tolist()
# =========================================================================
#                        p=normalize_position(any2direct(lattice_new, p), dtype='Direct')[:3]
# =========================================================================
                        p=any2direct(lattice_new, p)
                        if is_inside(position=p, selective_range=[[0, 1], [0, 1], [0, 1]], tolerance=2e-4):
                            positions.append(p[:3])
                            numbers.append(atom.element.z)
                        
        positions=np.array(positions)
        numbers=np.array(numbers)
        cell={'lattice':lattice_new, 'positions':positions, 'numbers':numbers}
        structure=Structure().create(raw_structure=cell2poscar(cell))
        
        structure.update(isPersist=isPersist)
        self._structure=structure
        return self
    
    def standardize(self, isUpdatedInfo=False, isPersist=False, **kwargs):
        """
        convert to standardized structure. 
        Note that if not specify the hall number, always the first one (the smallest serial number corresponding to 
        the space-group-type in list of space groups (Seto's web site)) among possible choices and settings is chosen as default.
        
        Arguments:
            isPersist (default=False): whether to save to the database.
            
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
                hall_number (default=0): hall number.
                
        Returns:
            structure's self (standardized).
        """
        import spglib
        from ..utils.fetch import get_symbol_by_z
        from ..materials.atom import Atom
        
        structure=self._structure
        
        cell=structure.formatting(dtype='cell')
        cell=(cell['lattice'], cell['positions'], cell['numbers'])
                    
        symprec=default_constants.symprec.value
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        hall_number=default_constants.hall_number.value
        if 'hall_nmber' in kwargs: hall_number=kwargs['hall_number']
        
        dataset=spglib.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=angle_tolerance, hall_number=hall_number)
        lattice_std=dataset['std_lattice']
        positions_std=dataset['std_positions']
        elements_std=np.array([get_symbol_by_z(z) for z in dataset['std_types']])
        
        # clear old atoms
        structure.withdraw()
        # update atoms
        structure.lattice=lattice_std
        for i in range(0, len(positions_std)):
            formated_atom=[elements_std[i]]+positions_std[i].tolist()+['Direct']
            structure.add_atom(atom=Atom().create(formated_atom=formated_atom))
        
        # default
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    def primitive(self, symprec=default_constants.symprec.value, isPersist=False):
        """
        primitive structure.
        
        Arguments:
            symprec (default=1e-5, symmetry tolerance): distance tolerance in Cartesian coordinates to find crystal symmetry.
            isPersist (default=False): whether to save to the database.
        
        Returns:
            structureFactory's object.
        """
        import spglib
        from ..utils.convert import cell2poscar
        from ..materials.structure import Structure
        
        structure=self._structure
        cell=self._structure.formatting('cell')
        cell=(cell['lattice'], cell['positions'], cell['numbers'])
        cell_new=spglib.find_primitive(cell, symprec=symprec)
        if cell_new is None: raise ValueError('The search is filed')
        
        cell_new={'lattice':cell_new[0], 'positions':cell_new[1], 'numbers':cell_new[2]}
        structure=Structure().create(raw_structure=cell2poscar(cell_new))
        
        structure.update(isPersist=isPersist)
        self._structure=structure
        return self
    
    def conventional(self, symprec=default_constants.symprec.value, isPersist=False):
        """
        conventional structure.
        
        Arguments:
            symprec (default=1e-5, symmetry tolerance): distance tolerance in Cartesian coordinates to find crystal symmetry.buj
            isPersist (default=False): whether to save to the database.
        
        Returns:
            structureFactory's object.
        """
        import spglib
        from ..utils.convert import cell2poscar
        from ..materials.structure import Structure
        
        structure=self._structure
        cell=self._structure.formatting('cell')
        cell=(cell['lattice'], cell['positions'], cell['numbers'])
        cell_new=spglib.standardize_cell(cell, symprec=symprec)
        if cell_new is None: raise ValueError('The search is filed')
        
        cell_new={'lattice':cell_new[0], 'positions':cell_new[1], 'numbers':cell_new[2]}
        structure=Structure().create(raw_structure=cell2poscar(cell_new))
        
        structure.update(isPersist=isPersist)
        self._structure=structure
        return self
    
    def supercell(self, dim, isPersist=False, **kwargs):
        """
        supercell structure.
        
        Arguments:
            dim: size of supercell. i.e. [2, 2, 2] (integral)
            isPersist (default=False): whether to save to the database.
            
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
        
        Returns:
            structureFactory's object.
        """
        from copy import deepcopy
        from ..materials.atom import Atom
        
        structure=self._structure
        
        # check
        if len(dim) != 3: raise ValueError('invalid dim')
        if False in [isinstance(s0, int) and s0 >= 1 for s0 in dim]: raise ValueError('invalid value in dim')
        
        # lattice
        for i in range(0, len(dim)):
            structure.lattice[i] *= dim[i]
         
        atoms=structure.atoms
        formated_atoms=[]
        for atom in atoms:
            formated_atoms.append(atom.to_formated_atom())
            
        # clear old atoms
        structure.withdraw()
        
        for formated_atom in formated_atoms:
            formated_atom0=deepcopy(formated_atom)
            for i in range(0, dim[0]): # x
                for j in range(0, dim[1]): # y
                    for k in range(0, dim[2]): # z
                        formated_atom0[1]=(formated_atom[1]+i)/dim[0] # x
                        formated_atom0[2]=(formated_atom[2]+j)/dim[1] # y
                        formated_atom0[3]=(formated_atom[3]+k)/dim[2] # z
                        structure.add_atom(Atom().create(formated_atom=formated_atom0), isUpdatedInfo=False, isPersist=False)

        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    def joint(self, jointed_structure, direction, isUpdatedInfo=False, isPersist=False, **kwargs):
        """
        joint two structures along given direction. Note that the size of the section perpendicular to the splicing direction 
            is equal to that of the structure, not the jointed structure.
        
        Arguments:
            jointed_structure: structure needed to joint.
            direction: direction vector to add the vacuum along lattice vector(a/b/c). The valid format is :
                [1, 0, 0, 'Direct'] # right of a
                [-1, 0, 0, 'Direct'] # left of a
            isUpdatedInfo (default=False): whether to update the composition and symmetry information (include the site, operation, wyckoffSite, spacegroup).
            isPersist (default=False): whether to save to the database.
            
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
        
        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    def cut(self, lattce_surface, isUpdatedInfo=False, isPersist=False, **kwargs):
        """
        cut along a lattice surface.
        """
        pass
    
    def mirror(self, atoms, mirror_plane, isUpdatedInfo=False, isPersist=False, **kwargs):
        """
        mirror given atoms along
        """
        pass
    
    def alloy(self):
        """
        """
        pass

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
    
    def surface(self, hkl, maxindex=None, saxis='c', isPersist=False, **kwargs):
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

        # Orient the uvw sets based on saxis 
        if saxis == 'c':
            uvws = np.array([a_uvw, b_uvw, c_uvw], dtype=int)
        elif saxis == 'b':
            uvws = np.array([b_uvw, c_uvw, a_uvw], dtype=int)
        elif saxis == 'a':
            uvws = np.array([c_uvw, a_uvw, b_uvw], dtype=int)

        return self.redefine(uvws, isPersist)
    
    def adsorption(self):
        """
        """
        pass

    def images(self, structure, numbers_of_images, isUpdatedInfo=False, isPersist=False, **kwargs):
        """
        
     
        """
        from ..materials.structure import Structure
        
        structure=self._structure
        
        # check
        if not isinstance(structure, Structure): 
            raise ValueError('unrecognized structure')
        
        images=[] # image structures
        
        return self

    def rotation(self, atoms, axis, theta, isUpdatedInfo=False, isPersist=False, **kwargs):
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
            isPersist (default=False): whether to save to the database.
            
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
        from ..utils.check import check_formated_position, check_formated_position_only_cartesian, check_formated_angle, check_formated_atom
        from ..utils.convert import any2direct, normalize_position, rotation
        from ..materials.atom import Atom
        from ..materials.structure import Structure
        from ..materials.molStructure import MolStructure
        
        structure=self._structure
        
        # check
        # axis
        if not check_formated_position(axis): raise ValueError('unrecognized axis')
        if isinstance(structure, Structure):
            axis=any2direct(structure.lattice, axis)
        elif isinstance(structure, MolStructure):
            if not check_formated_position_only_cartesian(axis): raise ValueError('unrecognized axis')
        # theta
        if not check_formated_angle(theta): raise ValueError('unrecognized theta')
        # origin
        origin=kwargs.get('origin', None)
        
# =============================================================================
#         # method 1
#         for atom in list(atoms):
#             formated_atom=None
#             if isinstance(atom, Atom):
#                 formated_atom=atom.to_formated_atom()
#             elif check_formated_atom(atom):
#                 formated_atom=atom
#             else:
#                 raise ValueError('unrecognized atom')
#                 
#             atom0=structure.get_atom(formated_atom)
#             atom0.position=normalize_position(rotation(atom0.position, axis, theta, origin=origin), dtype='d')[:-1]
# =============================================================================
        # method 2
        moving_atoms=[]
        for atom in list(atoms):
            formated_atom=None
            if isinstance(atom, Atom):
                formated_atom=atom.to_formated_atom()
            elif check_formated_atom(atom):
                formated_atom=atom
            else:
                raise ValueError('unrecognized atom')
            moving_atoms.append(structure.get_atom(formated_atom))
        for atom in moving_atoms:
            atom.position=normalize_position(rotation(atom.position, axis, theta, origin=origin), dtype='d')[:-1]
# =============================================================================
#         # method 3
#         moving_atoms=[]
#         for atom in list(atoms):
#             formated_atom=None
#             if isinstance(atom, Atom):
#                 formated_atom=atom.to_formated_atom()
#             elif check_formated_atom(atom):
#                 formated_atom=atom
#             else:
#                 raise ValueError('unrecognized atom')
#             atom0=structure.get_atom(formated_atom)
#             formated_atom2=[atom0.element.symbol]+normalize_position(rotation(atom0.position, axis, theta, origin=origin), dtype='d')[:-1] # new position
#             moving_atoms.append(formated_atom2)
#         self.del_atoms(atoms=atoms)
#         self.add_atoms(atoms=moving_atoms)
# =============================================================================

        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    def translation(self, atoms, direction, isUpdatedInfo=False, isPersist=False, **kwargs):
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
            isPersist (default=False): whether to save to the database.
            
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
        
        Returns:
            structureFactory's object.
        """
        from ..utils.check import check_formated_position, check_formated_atom
        from ..utils.convert import any2direct, normalize_position, translation
        from ..materials.atom import Atom
        
        structure=self._structure
        
        # check
        if not check_formated_position(direction): raise ValueError('invalid direction')
        direction=any2direct(structure.lattice, direction)
        
# =============================================================================
#         # method 1
#         for atom in list(atoms):
#             formated_atom=None
#             if isinstance(atom, Atom):
#                 formated_atom=atom.to_formated_atom()
#             elif check_formated_atom(atom):
#                 formated_atom=atom
#             else:
#                 raise ValueError('unrecognized atom')
#             atom0=structure.get_atom(formated_atom)
#             atom0.position=normalize_position(translation(atom0.position, direction), dtype='d')[:-1]
# =============================================================================
        # method 2
        moving_atoms=[]
        for atom in list(atoms):
            formated_atom=None
            if isinstance(atom, Atom):
                formated_atom=atom.to_formated_atom()
            elif check_formated_atom(atom):
                formated_atom=atom
            else:
                raise ValueError('unrecognized atom')
            moving_atoms.append(structure.get_atom(formated_atom))
        for atom in moving_atoms:
            atom.position=normalize_position(translation(atom.position, direction), dtype='d')[:-1]

# =============================================================================
#         # method 3
#         moving_atoms=[]
#         for atom in list(atoms):
#             formated_atom=None
#             if isinstance(atom, Atom):
#                 formated_atom=atom.to_formated_atom()
#             elif check_formated_atom(atom):
#                 formated_atom=atom
#             else:
#                 raise ValueError('unrecognized atom')
#             atom0=structure.get_atom(formated_atom)
#             formated_atom2=[atom0.element.symbol]+normalize_position(translation(atom0.position, direction), dtype='d')[:-1]
#             moving_atoms.append(formated_atom2)
#         self.del_atoms(atoms=atoms)
#         self.add_atoms(atoms=moving_atoms)
# =============================================================================
        
        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    # for Molecular dynamics
    def initializeVelocityDistribution(self, temperature, isUpdatedInfo=False, isPersist=False, **kwargs):
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
        
        velocities=np.random.random(3*structure.natoms)-0.5 # Angstrom/fs
        for i in range(0, structure.natoms):
            structure.atoms[i].velocity=velocities[i:i+3]
            momentum += structure.atoms[i].velocity
            ke += structure.atoms[i].element.mass*np.square(np.linalg.norm(structure.atoms[i].velocity))*1e7 # Kg x (m/s)^2
        momentum /= structure.natoms
        R=8.3144598 # Gas constant (J/Kmol)
        t0=ke/(R*3*(structure.natoms-1)) # calculated temperature
        scale=np.sqrt(temperature/t0)
        
        # scale to the desired temperature
        for atom in structure.atoms:
            atom.velocity=scale*(atom.velocity-momentum)
            
        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    def perturb(self, cutoff=0.1, isUpdatedInfo=False, isPersist=False, **kwargs):
        """
        perturb the atomic position.
        
        arguments:
            cutoff (default=0.1): cutoff of perturbation (unit: Angstrom). 
            
        """
        from ..utils.convert import any2cartesian, cartesian2direct
        
        structure=self._structure
        
        atoms=structure.atoms
        perturbations=np.random.randn(3*structure.natoms)*cutoff # unit: Angstrom
        
        # perturb positions
        for i in range(0, structure.natoms):
            position=any2cartesian(structure.lattice, atoms[i].position)
            for j in range(0, 3): 
                position[j] += perturbations[i+j]
            atoms[i].position=cartesian2direct(structure.lattice, position)[:3]
        
        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
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
        
    
    def removeUnit(self, unit, tolerance=0.1, isMoveAtoms=True, isUpdatedInfo=False, isPersist=False, **kwargs):
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
        
        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    def addUnit(self, unit, nrepeat, tolerance=0.1, isUpdatedInfo=False, isPersist=False, **kwargs):
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

        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    def insertMolecule(self, structure_of_molecule, position_in_molecule, position, isUpdatedInfo=False, isPersist=False, **kwargs):
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
            isPersist (default=False): whether to save to the database.
            
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
        
        Returns:
            structureFactory's object.
        """
        from ..utils.check import check_formated_position_only_direct, check_formated_position_only_cartesian
        from ..utils.convert import direct2cartesian
        
        structure=self._structure
        
        # check
# =========================================================================
#        # position_in_molecule
#        # if not check_formated_position_only_cartesian(position_in_molecule): raise ValueError('unknown position_in_molecule')
# =========================================================================

        # position
        if check_formated_position_only_direct(position):
            position=direct2cartesian(structure.lattice, position)
        elif check_formated_position_only_cartesian(position):
            pass
        else:
            raise ValueError('unknown position')
            
        direction=[0.0, 0.0, 0.0, 'Cartesian']
        for i in range(0, 3): 
            direction[i]=position[i]-position_in_molecule[i]
        
        atoms=structure_of_molecule.atoms
        formated_atoms=[]
        for a0 in atoms:
            a0=a0.to_formated_atom(dtype='Cartesian')
            for i in range(0, 3): a0[i+1] += direction[i]
            formated_atoms.append(a0)
        self.add_atoms(atoms=formated_atoms)
        
        # default
        symprec=default_constants.symprec.value # symprec
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value # angle_tolerance
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        if isUpdatedInfo: structure.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        self._structure=structure
        return self
    
    
    

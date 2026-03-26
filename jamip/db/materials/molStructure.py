# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from django.db  import models
import numpy as np

from .molComposition import MolComposition
from .molElement import MolElement
from .molAtom import MolAtom

from ..utils.variables import default_constants


class MolStructure(models.Model):
    """
    molecular structure.
    
    Note that:
        coordinate type of atomic positions can only be 'Cartesian' for molecule inside the jamip.
        
    Relationships:
        structure
            |- composition
            |- element
            |- atom
            
    Attributes:
        structure
            |- composition
            |- label
            |- natoms
            |- ntypes
            |- volume
            |- volume_per_atom
            # ---------- database ----------
            |- element_set
            |- atom_set
            # ---------- build-in ----------
            |- elements
            |- atoms
    """
    # relationship
    composition=models.ForeignKey('MolComposition', null=True, on_delete=models.PROTECT)
    element_set=models.ManyToManyField('MolElement')
    
    label=models.CharField(null=True, blank=True, max_length=80)
    
    natoms=models.IntegerField(blank=True, null=True)
    ntypes=models.IntegerField(blank=True, null=True)
    
    volume=models.FloatField(blank=True, null=True)
    volume_per_atom=models.FloatField(blank=True, null=True)
    
    class Meta:
        app_label='materials'
        db_table='molStructure'
        default_related_name='structure_set'
        
    def __str__(self):
        return self.composition.formula

    _elements=None
    @property
    def elements(self):
        """
        elements by retrieving atom.element.
        """
# =============================================================================
#         if self._elements is None:
#             self._elements=[]
#         return self._elements
# =============================================================================
        self._elements=[]
        if self.atoms != []:
            for atom in self.atoms:
                element=atom.element
                if not element in self._elements:
                    self._elements.append(element)
        return self._elements
                
    def get_element(self, symbol):
        """
        get the element's object with the given symbol.
        
        Arguments:
            symbol: element's symbol.
        
        Returns:
            element's object if it exists. Conversely, return the None.
        """
        for element in self.elements:
            if element.symbol == symbol:
                return element
        return None
    
    _atoms=None
    @property
    def atoms(self):
        """
        atoms contained this structure
        """
        if self._atoms is None:
            self._atoms=[]
        if self.id != None and self._atoms == []:
            self._atoms=self.atom_set.all()      # from DB by Django's Queryset object 
        return self._atoms

#    @profile
    def get_atom(self, formated_atom, **kwargs):
        """
        get the atom's object by list-type formated atom.
        
        Arguments:
            formated_atom: formated atom. Note that only have one type of coordinate is 'Cartesian for the molecule.
            the type. The valid formation:
                ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                
                Also, it is support the following format (Ignore element's symbol).
                [5.234, 0.0, 0.0, 'Cartesian']
                
            kwargs:
                precision (default=1e-3): used to determine whether the two atoms are overlapped. Note that, 
                        to determine whether this atom is in collection by comparing its distance 
                        from other atoms.
        Returns:
            atom's object if exist. Conversely, return None.
        """
        from ..utils.check import check_formated_atom_only_cartesian
        from ..utils.check import check_formated_position_only_cartesian
    
        precision=default_constants.precision.value
        if 'precision' in kwargs:
            precision=kwargs['precision']
        
        result=None
        position=None
        if check_formated_atom_only_cartesian(formated_atom):
                position=formated_atom[1:4] 
        elif check_formated_position_only_cartesian(formated_atom):
            position=formated_atom[:3]
        else: 
            import warnings
            warnings.warn('wrong format of formated_atom')
            return None

        for atom in self.atoms:
            distance=np.array(position)-atom.position                        
            if np.linalg.norm(distance) <= precision:
                if result is None:
                    result=atom
                else:
                    raise ValueError('exist overlap atoms with given position')
        return result
        
    def get_atoms_of_element(self, symbol):
        """
        get atoms of given element.
        
        Arguments:
            symbol: element's symbol.
        
        Return:
            list array of atoms.
        """
        atoms=[]
        for atom in self.atoms:
            if symbol == atom.element.symbol:
                atoms.append(atom)
        return atoms
    
    def add_atom(self, atom, isUpdatedInfo=False, isPersist=False, **kwargs):
        """
        add a atom to this structure.
        
        Arguments:
            atom: atom's object.
            
            kwargs:
                precision (default=1e-3): used to determine whether the two atoms are overlapped. Note that, 
                        to determine whether this atom is in collection by comparing its distance 
                        from other atoms.
                isCheckOverlap (default=Trule): check if the given atom will coincide with other atoms in this structure.
                        
        Return:
            structure's self.
        """
        precision=default_constants.precision.value
        if 'precision' in kwargs:
            precision=kwargs['precision']
        isCheckOverlap=True
        if 'isCheckOverlap' in kwargs: 
            isCheckOverlap=kwargs['isCheckOverlap']
        
        
        if not isinstance(atom, MolAtom):
            raise ValueError('not a instance of Atom')
        elif self.atoms is None:
            self.atoms=[atom]
        else:
            atom0=self.get_atom(formated_atom=atom.to_formated_atom(), precision=precision)
            if isCheckOverlap and not(atom0 is None):
                raise ValueError('exist position overlap in structure.atoms: {} -> {}'.format(str(atom.to_formated_atom()), str(atom0.to_formated_atom())))
            else:
                self.atoms.append(atom)
                atom.structure=self
                
        # update data
        if isUpdatedInfo:
            self.update(isPersist=isPersist)
                
        return self
    
    def del_atom(self, atom, isUpdatedInfo=False, isPersist=False):
        """
        remove a atom to this structure.
        
        Arguments:
            atom: atom's object.
            
        Return:
            structure's self.
        """
        if self.atoms is None:
            raise ValueError('structure.atoms is None')
        else:
            atom0=self.get_atom(formated_atom=atom.to_formated_atom())
            if not (atom0 in self.atoms):
                raise ValueError('not exist in structure.atoms: {}'.format(str(atom.to_formated_atom())))
            else:
                self.atoms.remove(atom)
                atom.structure=None
                
        # update data
        if isUpdatedInfo:
            self.update(isPersist=isPersist)
        
        return self
    
    def substitute_atom(self, atom, symbol_of_element, isUpdatedInfo=False, isPersist=False):
        """
        substitute the element of given atom in the structure.
        
        Arguments:
            atom: atom's object.
            symbol_of_element: new symbol of element.
            
        Return:
            structure's self.
        """
        from ..utils.check import check_symbol
        
        # check
        if not check_symbol(symbol_of_element):
            raise ValueError('unknown symbol_of_element_or_species')
        
        if self.atoms is None:
            raise ValueError('structure.atoms is None')
        else:
            formated_atom=atom.to_formated_atom()
            atom0=self.get_atom(formated_atom=formated_atom)
            if not (atom0 in self.atoms):
                raise ValueError('not exist in structure.atoms: {}'.format(str(atom.to_formated_atom())))
            else:
                formated_atom0=[symbol_of_element]+formated_atom[1:]
                self.del_atom(atom0)
                self.add_atom(MolAtom().create(formated_atom=formated_atom0))
        
        # update data
        if isUpdatedInfo:
            self.update(isPersist=isPersist)
        
        return self
    
    def __getElementsByAtoms(self):
        """
        Return:
            dictionary array.
        """
        elements={}
        for atom in self.atoms:
            element=atom.element
            # check
            if element is None:
                raise ValueError('atom.element is None')
                
            if not element.symbol in elements:
                elements[element.symbol]=1
            else:
                elements[element.symbol] += 1
        return elements
    
    def _getComposition(self):
        """
        compostion by atoms. Note that this method is different from that in Structure class.
        The reason is that there is not multiple relation in molecule owing to no translation periodicity.
        
        Return:
            formula: formula of composition.
            multi: number of formula in structure.
        """        
        elements=self.__getElementsByAtoms()
        
        element_symbols=list(elements.keys())
        numbers=list(elements.values())
        
        formula=None
        multi=None
        
        if not(numbers == []):
            formula=''
            for i in range(0, len(element_symbols)):
                if numbers[i] != 1:
                    formula += element_symbols[i]+str(int(numbers[i]))
                else:
                   formula += element_symbols[i]
# =============================================================================
#         if not(numbers == []):
#             multi=reduce(gcd, numbers) # number of formula
#         
#             formula=''
#             for i in range(0, len(element_symbols)):
#                 if int(numbers[i]/multi) != 1:
#                     formula += element_symbols[i]+str(int(numbers[i]/multi))
#                 else:
#                     formula += element_symbols[i]
# =============================================================================
        return formula, multi
    
    _mass=None
    @property
    def mass(self):
        """
        total atomic mass of this structure.
     
        Return:
            float value.
        """
# =============================================================================
#        self._mass=self.composition.mass
# =============================================================================

        atoms=self.atoms
        total_mass=np.sum([atom0.element.mass for atom0 in atoms])
        self._mass=total_mass
        return self._mass

    _mass_center=None
    @property
    def mass_center(self):
        """
        mass center of this structure.
        Return:
            float value.
        """        
        atoms=self.atoms
        
        sum4mr=0.0
        for atom0 in atoms:
            sum4mr += atom0.element.mass*atom0.position # sum of mi x r
        mass_center=sum4mr/self.mass
        self._mass_center=mass_center
        return self._mass_center

    def update(self, isPersist=False, **kwargs):
        """
        update data by atoms array in structure.
        
        Arguments:
            isPersist (default=False): whether to save to the database.
            
            kwargs:
                
                
        Return:
            structure's self.
        """
        # element
        # species
        # composition
        # spacegroup
        
        formula, multi=self._getComposition()
        
        if not(formula is None):
            # in Structure
            self.composition=MolComposition().create(formula=formula)
            self.natoms=len(self.atoms)
            self.ntypes=len(self.__getElementsByAtoms().keys())
            
            # in Composition
            self.composition._add_structure(self)
            for element in self.elements:
                self.composition._add_element(element)
            
            # in element
            for element in self.elements:
                element._add_structure(self)
                element._add_composition(self.composition)
                #element.add_atom
            
# =============================================================================
#             # in atom
#             for atom in self.atoms:
#                 atom.element.add_atom(atom) # element.atoms in element
#                 if not atom.species is None: atom.species.add_atom(atom) # species.atoms in species
# =============================================================================
            
            # check and remove
            # composition
            for composition in MolComposition.instances:
                if (composition.structures != []) and (self in composition.structures) and (not composition is self.composition):
                    composition.structures.remove(self)
            # element
            for element in MolElement.instances:
                if (element.structures != []) and (self in element.structures) and (not element in self.elements):
                    element.structures.remove(self)
                    
# =============================================================================
#             # some properties
#             self.volume=self.calculate_volume()
#             self.volume_per_atom=self.volume/self.natoms
# =============================================================================
        else: # nothing in structure
            self.composition=None
            self.natoms=0
            self.ntypes=0
            self.multiple=0
        
        if isPersist:
            self._persist()
            
    def _persist(self, **kwargs):
        """
        synchronize the data in database via data in memory. 
        
        Arguments:
            kwargs:
                precision (default=1e-3): used to determine whether the two atoms are overlapped. Note that, 
                        to determine whether this atom is in collection by comparing its distance 
                        from other atoms.
                        
        Return:
            structure's self.
        """
        from ..utils.check import is_overlap_of_positions_for_molecule
        
        if self.id is None: # didn't save into database
            # element
            for element in self.elements:
                element.save()
            # composition
            if not(self.composition is None): self.composition.save()
            # structure
            self.save()
            # atom
            for atom in self.atoms:
                atom.save()
            
            # relationship
            # structure
            for element in self.elements:
                self.element_set.add(element)
            for atom in self.atoms:
                self.atom_set.add(atom)
            # composition
            for structure in self.composition.structures:
                self.composition.structure_set.add(structure)
            for element in self.composition.elements:
                self.composition.element_set.add(element)
            # element
            for element in self.elements:
                for atom in self.get_atoms_of_element(element):
                    element.atom_set.add(atom)
        else:
            # structure
            # atom
            # delete rebundant atom from structure in database
            atoms_in_db=list(self.atom_set.all())
            for atom in atoms_in_db:
                if self.get_atom(formated_atom=atom.to_formated_atom()) is None: atom.delete()
            # add unsaved atom into structure in database
            precision=default_constants.precision.value
            if 'precision' in kwargs:
                precision=kwargs['precision']
            for atom0 in self.atoms:
                exist=False
                for atom_in_db0 in atoms_in_db:
                    if is_overlap_of_positions_for_molecule(formated_atom1=atom0.to_formated_atom(),
                                               formated_atom2=atom_in_db0.to_formated_atom(),
                                               precision=precision): exist=True
                if not exist:
                    atom0.element.save()
                    atom0.save()
                    self.atom_set.add(atom0)
            # element
            # delete rebundant element from structure in database
            elements_in_db=list(self.element_set.all())
            for element in elements_in_db:
#                if not(element.symbol in [e0.symbol for e0 in self.elements]):
                if not(element in self.elements): self.element_set.remove(element)
            # add unsaved element into structure in database
            for element0 in self.elements:
                if not(element0 in elements_in_db): self.element_set.add(element0)
            # composition
            self.composition.save()
            # structure
            # delete rebundant structure from composition in database
            structures_in_db=list(self.composition.structure_set.all())
# =============================================================================
#             for structure in structures_in_db:
#                 if not(structure in self.composition.structures): self.composition.structure_set.remove(structure)
# =============================================================================
            # add unsaved structure into composition in database
            for structure0 in self.composition.structures:
                if not(structure0 in structures_in_db): self.composition.structure_set.add(structure0)
            # element
            # delete rebundant element from composition in database
            elements_in_db=list(self.composition.element_set.all())
            for element in elements_in_db:
                if not(element in self.composition.elements): self.composition.element_set.remove(element)
            # add unsaved element into composition in database
            elements_in_db=list(self.composition.element_set.all())
            for element0 in self.composition.elements:
                if not(element0 in elements_in_db): self.composition.element_set.add(element0)

#    @profile
    def create(self, raw_structure, isPersist=False, **kwargs):
        """
        create a structure's object.
        Note that the type of atomic coordinate is Direct for crystal inside the code.
        
        Arguments:
            raw_structure: formated structure from Read method's returned dictionary.
            isPersist (default=False): whether to save to the database.
            
            kwargs:
                
        
        Returns:
            structure's self.
        """
        element_symbols=raw_structure['elements']
        numbers=raw_structure['numbers'] # atomic number of each element
        if len(element_symbols) != len(numbers):
            raise ValueError('inconsistent between elements and numbers')
        
        # label
        if 'label' in kwargs:
            self.label=kwargs['label']
            
        self.ntypes=len(element_symbols)
        self.natoms=np.sum(numbers)
        
# =============================================================================
#         self.volume=self.calculate_volume()
#         self.volume_per_atom=self.volume/self.natoms
# =============================================================================
        
        # dtype 
        dtype=raw_structure['type']
        if dtype == 'Direct':
            if 'lattice' not in raw_structure:
                raise ValueError('incomplete position type.')
        elif dtype != 'Cartesian':
            raise ValueError('Unknown position type.')

        # atom
        atom_index=0 # index of atom   

        for i in range(0, len(element_symbols)): # element
            for j in range(0, numbers[i]): # number of element
                symbol=element_symbols[i]
                position=raw_structure['positions'][atom_index]

                if dtype == 'Direct':
                    position = np.dot(position, raw_structure['lattice'])

                formated_atom=[symbol, position[0], position[1], position[2], 'Cartesian']

                atom=MolAtom().create(formated_atom=formated_atom,  
                                     isPersist=False)
                self.add_atom(atom)                                                            
                           
                atom_index += 1
            
        self.update(isPersist=isPersist)
        
        return self

    def formatting(self, dtype='xyz', **kwargs):
        """
        convert this object to a special formated object. 
        Note that the output structure is a standardized structure when dtypde is 'cif'. Perhaps, the output structure is different from the raw structure. 
        
        Arguments:
            dtype (default='xyz'): expected type. The supported type: xyz, mol.
            
            kwargs:
                for 'poscar' type:
                    coordinate_type:type of atomic coordinate (Direc/Cartesian).
                    isContainedConstraints: whether to contain the constraint information of atomic coordinate. 
                        The type of value is True or False.
                    isContainedVelocities: whether to contain the atomic velocity information (False/True).
                        
                for 'cif' type:
                    symprec (default=1e-5): precision when to find the symmetry.
                    angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
                    hall_number (default=0): hall number.
                    
        Returns:
            formated structure (python dictionary).
            
            for 'xyz' type:
                {'numbers':numbers,
                 'composition': formula_of_compositon
                 'atoms':formated_atoms}
                                  
            for 'mol' type:
                {'lattice':lattice,
                 'positions':positions,
                 'numbers':numbers,
                 'magmoms':magmoms(optional)}
        """
        formated_structure={}
        
        if dtype.strip().lower() == 'xyz':
            formated_structure['numbers']=self.natoms
            formated_structure['composition']=self.composition.formula
            
            formated_atoms=[]
            for atom0 in self.atoms: formated_atoms.append(atom0.to_formated_atom())
        
            formated_structure['atoms']=formated_atoms
        elif dtype.strip().lower() == 'mol':
            raise ValueError('need codes')
        else:
            raise ValueError('unrecognized dtype')
            
        return formated_structure

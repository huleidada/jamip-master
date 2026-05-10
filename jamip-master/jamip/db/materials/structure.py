# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from django.db  import models
import numpy as np
from ..utils.customField import NumpyArrayField
from ..utils.variables import default_constants

from .prototype import Prototype
from .entry import Entry
from .composition import Composition
from .element import Element
from .species import Species
from .atom import Atom
from .spacegroup import Spacegroup



class StructureError(Exception):
    pass

class Structure(models.Model):
    """
    crystal structure.
    Note that:
        1.the type of atomic position is 'Direct' inside the jamip.
        2.default, removing the translation periodicity. for example, 
            if atomic coordinate is 1.2, the program will change the 
            value to 0.2 in creating structure's instance.
            
    Relationships:
        structure
            |-prototype
            |-entry
            |-composition
            |-element
            |-species
            |-atom
            |-spacegroup
            
    Attributes:
        structure
            |-composition: composition of structure.
            |-spacegroup: specaegroup of structure.
            |-prototype: prototype of structure.
            |-label: label of structure.
            |-comment: comment of structure
            |-natoms: number of atoms in structure.
            |-nsites: number of sites in structure.
            |-ntypes: number of element's types in structure.
            |-multiple: number of formula in structure.
            |-lattice: lattice vector of structure. 
            |-volume: volume of lattice for the structure.
            |-volume_per_atom: volume of lattice per atom for the structure.
            # ---------- many-to-many in database ----------
            |-entry_set: cases contained the structure.
            |-element_set: elements of structure in database.
            |-species_set: species of structure in database.
            |-atom_set: atoms of structure in database.
            # ---------- build-in in memory ----------
            |-entries: collection of entries.
            |-elements: collection of elements in structure.
            |-species: collection of species in structure.
            |-atoms: collection of atoms in structure.
            |-site:
            |-wyckoffSite:
            
    Examples:
        >>> raw=Read('../examples/structures/In4Te3.cif').run()
        >>> s=Structure().create(raw)
    """

    # relationship
    composition=models.ForeignKey('Composition', null=True, on_delete=models.SET_NULL)
    element_set=models.ManyToManyField('Element', blank=True)
    species_set=models.ManyToManyField('Species', blank=True)
    #atom_set=models.ManyToManyField('Atom', blank=True)
    spacegroup=models.ForeignKey('Spacegroup', blank=True, null=True, on_delete=models.PROTECT)
    
    #prototype=models.ForeignKey('Prototype', blank=True, null=True, related_name='+')
    prototype=models.ForeignKey('Prototype', blank=True, null=True, on_delete=models.PROTECT)
    
    label=models.CharField(null=True, blank=True, max_length=80)
    comment=models.CharField(default='', max_length=80) # comment line in POSCAR (Line: 1)
    
    natoms=models.IntegerField(blank=True, null=True)
#    nsites=models.IntegerField(blank=True, null=True)
    ntypes=models.IntegerField(blank=True, null=True)
    
    multiple=models.IntegerField(blank=True, null=True) # times of formula
    
    lattice=NumpyArrayField(blank=True, null=True) # [3, 3] (float)
    volume=models.FloatField(blank=True, null=True)
    volume_per_atom=models.FloatField(blank=True, null=True)
    
    atomic_packing_factor=models.FloatField(blank=True, null=True)
    tolerance_factor=models.FloatField(blank=True, null=True)
    
    class Meta:
        app_label='materials'
        db_table='structure'
        default_related_name='structure_set'
        
    def __str__(self):
        #return self.composition.formula
        return 'Si'
        
    _entries=None
    @property
    def entries(self):
        """
        entries contained this structure
        """
        if self._entries is None:
            self._entries=[]
        return self._entries
    
    def add_entry(self, entry, isPersist=False):
        """
        add a entry to this structure.
        
        Arguments:
            entry: entry's object.
            
        Return:
            structure's self.
        """
        from .entry import Entry
        
        if not isinstance(entry, Entry):
            import warnings
            warnings.warn('not a instance of Entry')
        elif self.entries is None:
            self.entries=[entry]
            entry.structure=self
        else:
            if entry in self.entries:
                import warnings
                warnings.warn('exist in structure.entries')
            else:
                self.entries.append(entry)
                entry.structure=self
                
        # update data
        if isPersist: self.update(isPersist=isPersist)
            
        return self
    
    def del_entry(self, entry, isPersist=False):
        """
        remove a entry to this structure.
        
        Arguments:
            entry: entry's object.
            
        Return:
            structure's self.
        """
        if self.entries is None:
            import warnings
            warnings.warn('structure.entries is None')
        else:
            if not entry in self.entries:
                import warnings
                warnings.warn('not exist in structure.entries')
            else:
                self.entries.remove(entry)
                entry.structure=None
                
        # update data
        if isPersist: self.update(isPersist=isPersist)
            
        return self
    
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
            
    _species=None
    @property
    def species(self):
        """
        species by retrieving atom.species.
        """
        self._species=[]
        if self.atoms != []:
            for atom in self.atoms:
                species=atom.species
                if (not species is None) and (not species in self._species):
                    self._species.append(species)
        return self._species
    
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
            formated_atom: formated atom. Note that type of coordinate is 'Direct',  you can not specify
            the type. The valid formation:
                ['Na', 0.1, 0.0, 0.0, 'Direct']
                ['Na', 0.1, 0.0, 0.0] # Direct-type
                ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                
                contain species information:
                ['Na1+', 0.1, 0.0, 0.0, 'Direct']
                ['Na1+', 0.1, 0.0, 0.0]
                ['Na1+', 5.234, 0.0, 0.0, 'Cartesian']
                
                Also, it is support the following format (Ignore element's symbol).
                [0.1, 0.0, 0.0, 'Direct']
                [0.1, 0.0, 0.0] # Direct-type
                [5.234, 0.0, 0.0, 'Cartesian']
                
            kwargs:
                isNormalizingCoordinate (default=True): whether to remove the periodic boundary condition, 
                    ensure the value of atomic coordinate is between 0 and 1 (i.e. 1.3 -> 0.3).
                precision (default=1e-3): used to determine whether the two atoms are overlapped. Note that, 
                        to determine whether this atom is in collection by comparing its distance 
                        from other atoms.
        Returns:
            atom's object if exist. Conversely, return None.
        """
        from ..utils.check import check_formated_atom_only_direct, check_formated_atom_only_cartesian
        from ..utils.check import check_formated_position_only_direct, check_formated_position_only_cartesian
        from ..utils.convert import any2direct, normalize_position
    
        # remove atomic translation periodicity
#        isNormalizingCoordinate=True
        isNormalizingCoordinate=default_constants.isNormalizingCoordinate.value
        if 'isNormalizingCoordinate' in kwargs: isNormalizingCoordinate=kwargs['isNormalizingCoordinate']
        precision=default_constants.precision.value
        if 'precision' in kwargs: precision=kwargs['precision']
        
        result=None
        position=None
        if check_formated_atom_only_direct(formated_atom):
                position=formated_atom[1:]
        elif check_formated_atom_only_cartesian(formated_atom):
                position=any2direct(self.lattice, formated_atom[1:])   
        elif check_formated_position_only_direct(formated_atom):
            position=formated_atom
        elif check_formated_position_only_cartesian(formated_atom):
            position=any2direct(self.lattice, formated_atom)
        else: 
            import warnings
            warnings.warn('wrong format of formated_atom')
            return None
        if isNormalizingCoordinate:
            position=normalize_position(position, 'Direct')[:-1] # i.g. [0.1, 0.0, 0.0]
        for atom in self.atoms:
            distance=np.array(position)-atom.position
            
            # note the periodic boundary condition. like [0.0, 0.0, 0.0] vs. [0.999999, 0.0, 0.0]
            for j in range(0, len(distance)):
                if distance[j] > 0.5:
                    distance[j]=1-distance[j]
                elif distance[j] < -0.5:
                    distance[j]=1+distance[j]
                        
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
    
    def get_atoms_of_species(self, name):
        """
        get atoms of given element.
        
        Arguments:
            name: species's name.
        
        Return:
            list array of atoms.
        """
        atoms=[]
        for atom in self.atoms:
            if name == atom.species.name:
                atoms.append(atom)
        return atoms
    
    def add_atom(self, atom, isUpdatedInfo=False, isPersist=False, isCheckOverlap=False, **kwargs):
        """
        add a atom to this structure.
        
        Arguments:
            atom: atom's object.
            
            kwargs:
                isNormalizingCoordinate (default=True): whether to remove the periodic boundary condition, 
                    ensure the value of atomic coordinate is between 0 and 1 (i.e. 1.3 -> 0.3).
                precision (default=1e-3): used to determine whether the two atoms are overlapped. Note that, 
                        to determine whether this atom is in collection by comparing its distance 
                        from other atoms.
                isCheckOverlap (default=Trule): check if the given atom will coincide with other atoms in this structure.
                        
        Return:
            structure's self.
        """
        from .atom import Atom
        from ..utils.convert import normalize_position
        
        # remove atomic translation periodicity
        isNormalizingCoordinate=kwargs.get('isNormalizingCoordinate', default_constants.isNormalizingCoordinate.value)
        precision=kwargs.get('precision', default_constants.precision.value)
        
        if not isinstance(atom, Atom):
            raise ValueError('not a instance of Atom')
        elif self.atoms is None:
            self.atoms=[atom]
        else:
            if isNormalizingCoordinate:
                position=atom.position
                atom.position=normalize_position(position, 'Direct')[:-1] # i.g. [1.1, 0.0, 0.0] -> [0.1, 0.0, 0.0]
            
            if isCheckOverlap:
                atom0=self.get_atom(formated_atom=atom.position, precision=precision)
                if not (atom0 is None):
                    raise ValueError('exist position overlap in structure.atoms: {} -> {}'\
                        .format(str(atom.to_formated_atom()), str(atom0.to_formated_atom())))
            else:
                self.atoms.append(atom)
                atom.structure=self
                
        # update data
        if isUpdatedInfo: self.update(isPersist=isPersist)
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
# =============================================================================
#             import warnings
#             warnings.warn('structure.atoms is None')
# =============================================================================
            raise ValueError('structure.atoms is None')
        else:
            atom0=self.get_atom(formated_atom=atom.to_formated_atom())
            if not (atom0 in self.atoms):
# =============================================================================
#                 import warnings
#                 warnings.warn('not exist in structure.atoms: {}'.format(str(atom.to_formated_atom())))
# =============================================================================
                raise ValueError('not exist in structure.atoms: {}'.format(str(atom.to_formated_atom())))
            else:
                self.atoms.remove(atom0)
                atom.structure=None
                
        # update data
        if isUpdatedInfo: self.update(isPersist=isPersist)
        return self
    
    def substitute_atom(self, atom, symbol_of_element_or_species, isUpdatedInfo=False, isPersist=False):
        """
        substitute the element of given atom in the structure.
        
        Arguments:
            atom: atom's object.
            symbol_of_element_or_species: new symbol of element.
            
        Return:
            structure's self.
        """
        from ..utils.check import check_symbol, check_species
        
        # check
        if not(check_symbol(symbol_of_element_or_species)) and not(check_species(symbol_of_element_or_species)):
            raise ValueError('unknown symbol_of_element_or_species')
        
        if self.atoms is None:
            raise ValueError('structure.atoms is None')
        else:
            formated_atom=atom.to_formated_atom()

            atom0=self.get_atom(formated_atom=formated_atom)
            if not (atom0 in self.atoms):
                raise ValueError('not exist in structure.atoms: {}'.format(str(atom.to_formated_atom())))
            else:
                formated_atom0=[symbol_of_element_or_species]+formated_atom[1:]
                self.del_atom(atom0)
                self.add_atom(Atom().create(formated_atom=formated_atom0))
        
        # update data
        if isUpdatedInfo: self.update(isPersist=isPersist)
        return self
                
    _sites=None
    @property
    def sites(self):
        """
        sites contained this structure
        
        Return:
            list array. [index0, position0(1x3),
                         index1, position1(1x3),
                         ...]
            Note that: index is the index of corresponding atom in structure.
        """
# =============================================================================
#         if self._sites is None:
#             self._sites=[]
#         return self._sites
# =============================================================================
        return self._findSymmetryInfo()['sites']
        
    def get_site(self, atom):
        """
        get iequivalent site by given atom.
        
        Arguments:
            atom: atom's object.
            
        Return:
            position of site. i.e. [0.1, 0, 0]
        """
        site=None
        index_of_atom=self.atoms.index(atom)
        maps_of_sites=self._findSymmetryInfo()['map_of_sites']
        for key, value in maps_of_sites.items():
            if  index_of_atom in value: site=self.atoms[key].position
        return site
        
    def get_all_equivalent_atoms(self, atom):
        """
        get all equivalent atom by given atom.
        
        Arguments:
            atom: atom's object.
            
        Return:
            list of atoms.
        """
        atoms=None
        index_of_atom=self.atoms.index(atom)
        maps_of_sites=self._findSymmetryInfo()['map_of_sites']
        for key, value in maps_of_sites.items():
            if  index_of_atom in value: atoms=[self.atoms[index] for index in value]
        return atoms
    
    _wyckoffSites=None
    @property
    def wyckoffSites(self):
        """
        wyckoffSites contained this structure
        
        Return:
            list array
        """
        return self._findSymmetryInfo()['wyckoffs']
    def get_wyckoffSymbol(self, atom):
        """
        get wyckoff's symbol by given atom.
        
        Arguments:
            atom: atom's object.
        
        Return:
            wyckoff symbol.
        """
        return self._get_wyckoff(atom)['symbol']
    def get_wyckoffInfo(self, atom, dtype='symbol'):
        """
        get wyckoff's multiplicity by given atom.
        
        Arguments:
            atom: atom's object.
            dtype (default='symbol'):  type of output. 'symbol' or 'multiplicity' or 'position'
            
        Return:
            wyckoff symbol.
        """
        info=None
        if dtype.lower().startswith('s'):
            info=self._get_wyckoff(atom)['symbol']
        elif dtype.lower().startswith('m'):
            info=self._get_wyckoff(atom)['multiplicity']
        elif dtype.lower().startswith('p'):
            info=self._get_wyckoff(atom)['position']
        else:
            import warnings
            warnings.warn('unkonw dtype')
        return info
    
    def _get_wyckoff(self, atom):
        """
        get wyckoff by given atom.
        
        Arguments:
            atom: atom's object.
        
        Return:
            dictionary of wyckoff.
        """
        wyckoff=None
        info=self._findSymmetryInfo()
        index_of_atom=self.atoms.index(atom)
        maps_of_sites=info['map_of_sites']
        wyckoffs=info['wyckoffs']
        for key, value in maps_of_sites.items():
            if index_of_atom in value: wyckoff=wyckoffs[key]
        return wyckoff
    
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
        compostion by atoms.
        
        Return:
            formula: formula of composition.
            multi: number of formula in structure.
        """
        #from fractions import gcd
        #from functools import reduce
        
        elements=self.__getElementsByAtoms()
        
        element_symbols=list(elements.keys())
        numbers=list(elements.values())
        
        formula=None
        multi=None
        if not(numbers == []):
            #multi=reduce(gcd, numbers) # number of formula
            multi = np.gcd.reduce(numbers) # number of formula
        
            formula=''
            for i in range(0, len(element_symbols)):
                if int(numbers[i]/multi) != 1:
                    formula += element_symbols[i]+str(int(numbers[i]/multi))
                else:
                    formula += element_symbols[i]
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
#      self._mass=self.composition.mass*self.multiple
# =============================================================================
        atoms=self.atoms
        total_mass=np.sum([atom0.element.mass for atom0 in atoms])
        self._mass=total_mass
        return self._mass       

    _mass_center = None
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
            sum4mr += atom0.element.mass*np.array(atom0.position) # sum of mi x r
        mass_center=sum4mr/self.mass
        self._mass_center=mass_center
        return self._mass_center
    
    def update(self, isPersist=False, **kwargs):
        """
        update data by atoms array in structure.
        
        Arguments:
            isPersist (default=False): whether to save to the database.
            
            kwargs:
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
                
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
            self.composition=Composition().create(formula=formula)
            self.natoms=len(self.atoms)
            self.ntypes=len(self.__getElementsByAtoms().keys())
            self.multiple=multi
            
            symprec=default_constants.symprec.value
            if 'symprec' in kwargs: symprec=kwargs['symprec']
            angle_tolerance=default_constants.angle_tolerance.value
            if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
            
            symmetry_info=self._findSymmetryInfo(symprec=symprec, angle_tolerance=angle_tolerance)
            self.spacegroup=Spacegroup().create(number=symmetry_info['number'], 
                                      international=symmetry_info['international'],
                                      hm=symmetry_info['hall_number'],
                                      hall=symmetry_info['hall'],
                                      pointgroup=symmetry_info['pointgroup'])
            
            # in prototype
            # if not self.prototype is None: self.prototype._add_structure(self) 
            
            # # in Composition
            # self.composition._add_prototype(self.prototype)
            # self.composition._add_structure(self)
            # for element in self.elements:
            #     self.composition._add_element(element)
            
            # # in element
            # for element in self.elements:
            #     element._add_structure(self)
            #     element._add_composition(self.composition)
            
            # # in species
            # for species in self.species:
            #     species.element._add_species(species) # element.species in element                
            #     species._add_structure(self)
            
            # # in spacegroup
            # self.spacegroup._add_structure(self)
            
# =============================================================================
#             # in atom
#             for atom in self.atoms:
#                 atom.element.add_atom(atom) # element.atoms in element
#                 if not atom.species is None: atom.species.add_atom(atom) # species.atoms in species
# =============================================================================
            
            # check and remove
            # prototype
            # for prototype in Prototype.instances:
            #     if (prototype.structures != []) and (self in prototype.structures) and (not prototype is self.prototype):
            #         prototype.structures.remove(self)
            # # composition
            # for composition in Composition.instances:
            #     if (composition.structures != []) and (self in composition.structures) and (not composition is self.composition):
            #         composition.structures.remove(self)
            # # element
            # for element in Element.instances:
            #     if (element.structures != []) and (self in element.structures) and (not element in self.elements):
            #         element.structures.remove(self)
            # # species
            # for species in Species.instances:
            #     if (species.structures != []) and (self in species.structures) and (not species in self.species):
            #         species.structures.remove(self)
            # # spacegroup
            # for spacegroup in Spacegroup.instances:
            #     if (spacegroup.structures != []) and (self in spacegroup.structures) and (not spacegroup is self.spacegroup):
            #         spacegroup.structures.remove(self)
                    
            # some properties
            self.volume=self.calculate_volume()
            self.volume_per_atom=self.volume/self.natoms
            
# =============================================================================
#             self.atomic_packing_factor=
#             self.tolerance_factor=
# =============================================================================
            
        else: # nothing in structure
            self.composition=None
            self.natoms=0
            self.ntypes=0
            self.multiple=0
            self.spacegroup=None
        
        if isPersist: self._persist()
            
# =============================================================================
#     _history=None # {time: operation, ...}
#     @property
#     def history(self):
#         """
#         record historical operations.
#         """
#         if self._history is None:
#             self._history={}
#         return self._history
#     
#     def __add_history(self, dtype, **kwargs):
#         """
#         add historical operation.
#         
#         Arguments:
#             dtype: type of operation. 'add_entry', 'del_entry', 'add_atom', 'del_atom', 'create', 'withdraw'
#             
#             kwargs:
#                 object_of_operation: object of operation.
#                     for 'add_entry', 'del_entry', 'add_atom', 'del_atom':
#                         object of entry or atom
#                     for 'create', 'withdraw':
#                         None
#                 
#         Return:
#             history's self
#         """
#         from utils.fetch import get_time
#         
#         if dtype.lower() == 'add_entry' or 'ae':
#             self.history[get_time()]=kwargs['object_of_operation']
#         elif dtype.lower() == 'del_entry' or 'de':
#             self.history[get_time()]=kwargs['object_of_operation']
#         elif dtype.lower() == 'add_atom' or 'aa':
#             self.history[get_time()]=kwargs['object_of_operation']
#         elif dtype.lower() == 'del_atom' or 'da':
#             self.history[get_time()]=kwargs['object_of_operation']
#         elif dtype.lower().startswith('c'):
#             self.history[get_time()]=None
#         elif dtype.lower().startswith('w'):
#             self.history[get_time()]=None
#         else:
#             raise ValueError('unknown dtype')
# =============================================================================
    
    def _persist(self, **kwargs):
        """
        synchronize the data in database via data in memory. 
        
        Arguments:
            kwargs:
                isNormalizingCoordinate (default=True): whether to remove the periodic boundary condition, 
                    ensure the value of atomic coordinate is between 0 and 1 (i.e. 1.3 -> 0.3).
                precision (default=1e-3): used to determine whether the two atoms are overlapped. Note that, 
                        to determine whether this atom is in collection by comparing its distance 
                        from other atoms.
        
        Return:
            structure's self.
        """
        from ..utils.check import is_overlap_of_positions
        
        if self.id is None: # didn't save into database
            # spacegroup
            if not(self.spacegroup is None): self.spacegroup.save()
            # element
            for element in self.elements:
                element.save()
            # species
            for species in self.species:
                species.save()
            # composition
            if not(self.composition is None): self.composition.save()
            # structure
            self.save()
            # atom
            for atom in self.atoms:
                atom.save()
            # entry
            for entry in self.entries:
                entry.save()
            # prototype
            if not(self.prototype is None): self.prototype.save()
            
            # relationship
            # structure
            for element in self.elements:
                self.element_set.add(element)
            for species in self.species:
                self.species_set.add(species)
            for atom in self.atoms:
                self.atom_set.add(atom)
            for entry in self.entries:
                self.entry_set.add(entry)
            # composition
            for prototype in self.composition.prototypes:
                self.composition.prototype_set.add(prototype)
            for structure in self.composition.structures:
                self.composition.structure_set.add(structure)
            for element in self.composition.elements:
                self.composition.element_set.add(element)
            # element
            for element in self.elements:
                for species in element.species:
                    element.species_set.add(species)
                for atom in self.get_atoms_of_element(element):
                    element.atom_set.add(atom)
            
        else:
            # spacegroup
            if Structure.objects.get(id=self.id).spacegroup.number != self.spacegroup.number:
                self.spacegroup.save()
                self.spacegroup.structure_set.add(self)
            
            # structure
            # atom
            # delete rebundant atom from structure in database
            atoms_in_db=list(self.atom_set.all())
            for atom in atoms_in_db:
                if self.get_atom(formated_atom=atom.to_formated_atom()) is None: atom.delete()
            # add unsaved atom into structure in database
            isNormalizingCoordinate=default_constants.isNormalizingCoordinate.value
            if 'isNormalizingCoordinate' in kwargs: isNormalizingCoordinate=kwargs['isNormalizingCoordinate']
            precision=default_constants.precision.value
            if 'precision' in kwargs: precision=kwargs['precision']
            
            for atom0 in self.atoms:
                exist=False
                for atom_in_db0 in atoms_in_db:
                    if is_overlap_of_positions(formated_atom1=atom0.to_formated_atom(),
                                               formated_atom2=atom_in_db0.to_formated_atom(),
                                               lattice=self.lattice,
                                               precision=precision,
                                               isNormalizingCoordinate=isNormalizingCoordinate): exist=True
                if not exist:
                    atom0.element.save()
                    if not(atom0.species is None): atom0.species.save()
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
            # species
            # delete rebundant species from structure in database
            species_in_db=list(self.species_set.all())
            for species in species_in_db:
                if not(species in self.species): self.species_set.remove(species)
            # add unsaved species into structure in database
            for species0 in self.species:
                if not(species0 in species_in_db): self.species_set.add(species0)
            
            # entry
            # delete rebundant entries from structure in database
            entries_in_db=list(self.entry_set.all())
            for entry in entries_in_db:
                if not(entry in self.entries): self.entry_set.remove(species)
            # add unsaved entry into structure in database
            for entry0 in self.entries:
                if not(entry0 in entries_in_db):
                    entry0.save()
                    self.entry_set.add(entry0)
           
            # composition
            self.composition.save()
            # prototype
            # delete rebundant prototype from composition in database
            prototypes_in_db=list(self.composition.prototype_set.all())
# =============================================================================
#             for prototype in prototypes_in_db:
#                 if not(prototype in self.composition.prototypes): self.composition.prototype_set.remove(prototype)
# =============================================================================
            # add unsaved prototype into composition in database
            for prototype0 in self.composition.prototypes:
                if not(prototype0 in prototypes_in_db):
                    prototype0.save()
                    self.composition.prototype_set.add(prototype0)
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
                isNormalizingCoordinate (default=True): whether to remove the periodic boundary condition, 
                    ensure the value of atomic coordinate is between 0 and 1 (i.e. 1.3 -> 0.3). 
                    
                isContainedConstraints (default=False): whether to write the constrain information to this structure when raw_structure contain them.
                isContainedVelocities (default=False): whether to write the velocity information to this structure when raw_structure contain them.
                   
                symprec (default=1e-5): precision when to find the symmetry.
                angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
                
                # ---------- composition ----------
                generic:
                
                # ---------- atom --------
                species: collection of species's name for each atom. i.e. ['Fe2+', 'Fe3+', 'Sb-',...]
        
        Returns:
            structure's self.
        """
        from ..utils.check import check_species
        from ..utils.fetch import get_symbol_from_species
            
        element_symbols=raw_structure['elements']
        numbers=raw_structure['numbers'] # atomic number of each element
        if len(element_symbols) != len(numbers): raise StructureError('inconsistent between elements and numbers')
        
        generic=None
        if 'generic' in kwargs: generic=kwargs[generic]
        
        # label
        if 'label' in kwargs: self.label=kwargs['label']
        
        if 'comment' in raw_structure: self.comment=raw_structure['comment']
            
        self.ntypes=len(element_symbols)
        self.natoms=np.sum(numbers)
        
        self.lattice=raw_structure['lattice']
        self.volume=self.calculate_volume()
        self.volume_per_atom=self.volume/self.natoms
        
        # atom
        # remove translation periodicity
        isNormalizingCoordinate=default_constants.isNormalizingCoordinate.value
        if 'isNormalizingCoordinate' in kwargs: isNormalizingCoordinate=kwargs['isNormalizingCoordinate']
        
        # constrain    
        constraints=None
        if 'constraints' in raw_structure: constraints=raw_structure['constraints']
        isContainedConstraints=default_constants.isContainedConstraints.value
        if ('isContainedConstraints' in kwargs) and kwargs['isContainedConstraints']:
            isContainedConstraints=kwargs['isContainedConstraints']
            if not isinstance(isContainedConstraints, bool): raise ValueError("unrecognized value of 'isContainedConstraints'")
            
        # velocity
        velocities=None
        if 'velocities' in raw_structure: velocities=raw_structure['velocities']
        isContainedVelocities=default_constants.isContainedVelocities.value
        if ('isContainedVelocities' in kwargs) and kwargs['isContainedVelocities']:
            isContainedVelocities=kwargs['isContainedVelocities']
            if not isinstance(isContainedVelocities, bool): raise ValueError("unrecognized value of 'isContainedVelocities'")

        raw_species=None
        if 'species' in kwargs:
            raw_species=kwargs['species']
            if len(raw_species) != self.natoms: raise ValueError('inconsistent dimensions between species and atoms')

        atom_index=0 # index of atom   
        for i in range(0, len(element_symbols)): # element
            for j in range(0, numbers[i]): # number of element
                symbol=element_symbols[i]
                position=raw_structure['positions'][atom_index]
                dtype=raw_structure['type']
                
                formated_atom=None
                if raw_species is None:
                    formated_atom=[symbol, position[0], position[1], position[2], dtype]
                else:
                    # check
                    species0=raw_species[atom_index]
                    if check_species(species0) and (get_symbol_from_species(species0) == symbol):
                        formated_atom=[species0, position[0], position[1], position[2], dtype]
                    else:
                        raise ValueError('unknown species in raw_species')
                    
                atom=Atom().create(formated_atom=formated_atom, 
                                     isNormalizingCoordinate=isNormalizingCoordinate, 
                                     lattice=self.lattice,
                                     isPersist=False)
                self.add_atom(atom)                                                            
                    
                if isContainedConstraints and ((not constraints is None) or constraints != []):
                    atom.constraint=constraints[atom_index]
                
                if isContainedVelocities and ((not velocities is None) or velocities != []):
                    atom.velocity=velocities[atom_index]
                           
                atom_index += 1
                
        symprec=default_constants.symprec.value
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']  
            
        self.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        return self
    
# =============================================================================
#     def standardize(self, isPersist=False, **kwargs):
#         """
#         convert to standardized structure. 
#         Note that if not specify the hall number, always the first one (the smallest serial number corresponding to 
#         the space-group-type in list of space groups (Seto’s web site)) among possible choices and settings is chosen as default.
#         
#         Arguments:
#             isPersist (default=False): whether to save to the database.
#             
#             kwargs:
#                 symprec (default=1e-5): precision when to find the symmetry.
#                 angle_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
#                 hall_number (default=0): hall number.
#                 
#         Returns:
#             structure's self (standardized).
#         """
#         import spglib
#         from utils.convert import raw2std_position
#         
#         cell=self.formatting(dtype='cell')
#         cell=(cell['lattice'], cell['positions'], cell['numbers'])
#                     
#         symprec=default_constants.symprec.value
#         if 'symprec' in kwargs:
#             symprec=kwargs['symprec']
#         angle_tolerance=default_constants.angle_tolerance.value
#         if 'angle_tolerance' in kwargs:
#             angle_tolerance=kwargs['angle_tolerance']
#             
#         hall_number=default_constants.hall_number.value
#         if 'hall_nmber' in kwargs:
#             hall_number=kwargs['hall_number']
#         dataset=spglib.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=angle_tolerance, hall_number=hall_number)
#         lattice_std=dataset['std_lattice']
#         
#         self.lattice=lattice_std
#         
#         for atom in self.atoms:
#             position=atom.position
#             atom.position=raw2std_position(position, dataset['transformation_matrix'], dataset['origin_shift'])
#         
#         self.update(isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
#         
#         self.spacegroup.hall=dataset['hall']
#         self.spacegroup.hm=dataset['hall_number']
#         self.spacegroup.save()
#         
#         return self
# =============================================================================
    
    def clone(self, isPersist=False, **kwargs):
        """
        clone a structure.
        
        Arguments:
            isPersist (default=False): whether to save to the database.
            
            kwrages:
                precision (default=1e-3): used to determine whether the two positions are overlapped.
                    from other atoms.
        Return:
            cloned structure's object.
        """
        import copy
        
        isNormalizingCoordinate=default_constants.isNormalizingCoordinate.value
        if 'isNormalizingCoordinate' in kwargs: isNormalizingCoordinate=kwargs['isNormalizingCoordinate']
        precision=default_constants.precision.value
        if 'precision' in kwargs: precision=kwargs['precision']
        
        structure=self.minimize_clone()
        
        # structure
        structure.label=self.label
        structure.comment=self.comment
        # species
        for atom in self.atoms:
            if not(atom.species is None):
                structure.get_atom(formated_atom=atom.to_formated_atom(), 
                                   isNormalizingCoordinate=isNormalizingCoordinate, 
                                   precision=precision).species=atom.species
        # entry
        for entry in self.entries:
            structure.add_entry(copy.copy(entry))
        # prototype
        structure.prototype=self.prototype
        
        symprec=default_constants.symprec.value
        if 'symprec' in kwargs: symprec=kwargs['symprec']
        angle_tolerance=default_constants.angle_tolerance.value
        if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
        
        if isPersist: self.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
        return structure
        
    
    def minimize_clone(self, isPersist=False):
        """
        clone a structure contains only the most basis information.
        
        Arguments:
            isPersist (default=False): whether to save to the database.
            
        Reurn:
            cloned structure's object.
        """
        return Structure().create(raw_structure=self.formatting(dtype='poscar'), 
                         isPersist=isPersist)
        
    
    def withdraw(self, isPersist=False):
        """
        destroy object.
        
        Arguments:
            isPersist (default=False): whether to save to the database.
            
        Return:
            Null.
        """
        for atom in list(self.atoms):
            self.del_atom(atom)
        self.update(isPersist=isPersist)
    
    def calculate_volume(self):
        """
        calculate the volume of this structure.
        
        Returns:
           calculated volume (float).
        """
        lattice=self.lattice
        return np.linalg.det(lattice)
    
    @property
    def lattice_parameters(self):
        """
        Return:
            lattice parameters [a, b, c, alpha, beta, gamma].
        """
        a=np.linalg.norm(self.lattice[0])
        b=np.linalg.norm(self.lattice[1])
        c=np.linalg.norm(self.lattice[2])
        alpha=np.degrees(np.arccos(np.clip(np.dot(self.lattice[1]/b, self.lattice[2]/c), -1, 1)))
        beta=np.degrees(np.arccos(np.clip(np.dot(self.lattice[0]/a, self.lattice[2]/c), -1, 1)))
        gamma=np.degrees(np.arccos(np.clip(np.dot(self.lattice[0]/a, self.lattice[1]/b), -1, 1)))
        
        return np.array([a, b, c, alpha, beta, gamma])
        
    def formatting(self, dtype='poscar', **kwargs):
        """
        convert this object to a special formated object. 
        Note that the output structure is a standardized structure when dtypde is 'cif'. Perhaps, the output structure is different from the raw structure. 
        
        Arguments:
            dtype (default='poscar'): expected type. The supported type: cif, poscar, cell.
            
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
            
            for 'poscar' type:
                {'comment':comment,
                 'lattice':lattice,
                 'elements':elements,
                 'numbers':numbers,
                 'type':type,
                 'positions':positions,
                 'constraints':constraints(optional)}
                 
            for 'cif' type:
                {'data_':formula,
                 '_cell_length_a':crya,
                 '_cell_length_b':cryb,
                 '_cell_length_c':cryc,
                 '_cell_angle_alpha':alpha,
                 '_cell_angle_beta':beta,
                 '_cell_angle_gamma':gamma,
                 '_cell_volume':cellvolume,
                 '_symmetry_space_group_name_H-M':hmsyb,
                 '_symmetry_Int_Tables_number':itnum,
                 '_symmetry_equiv_pos_as_xyz':symopt,
                 '_atom_site_label':sitelab,
                 '_atom_site_wyckoff_symbol':wyc,
                 '_atom_site_fract_x':sitex,
                 '_atom_site_fract_y':sitey,
                 '_atom_site_fract_z':sitez,
                 '_atom_site_occupancy':occu}
                 
            for 'cell' type:
                {'lattice':lattice,
                 'positions':positions,
                 'numbers':numbers,
                 'magmoms':magmoms(optional)}
        """
        from copy import deepcopy
        from ..iostream.cif import format_symbol
        #from ..iostream.spaceGroupD3 import spacegroups as sgd
        #from ..iostream.hall2hm import hl2hm as hm
        from ..iostream.symgroup import SpaceGroup
        from ..utils.convert import any2cartesian
        from ..modeling.structureFactory import StructureFactory
        
        if dtype.strip().lower() == 'poscar':
            comment=self.comment
            lattice=deepcopy(self.lattice)
            
            coordinate_type='Direct'
            if 'coordinate_type' in kwargs:
                if kwargs['coordinate_type'].strip().lower().startswith('c'):
                    coordinate_type='Cartesian'
                elif kwargs['coordinate_type'].strip().lower().startswith('d'):
                    coordinate_type='Direct'
                else:
                    raise StructureError("unrecognized type of atomic coordinate (Direct/Cartesian)")
                
            # elements, numbers, position, constrain (optional) and velocities (optional)
            elements=[]
            numbers=[]
            positions=[]
            constraints=[]
            velocities=[]
            for i in range(0, len(self.elements)):
                elements.append(self.elements[i].symbol)
                atoms_of_element=self.get_atoms_of_element(elements[i])
                numbers.append(len(atoms_of_element)) # need to check carefully
                for j in range(0, numbers[i]):
                    if coordinate_type == 'Direct':
                        positions.append(atoms_of_element[j].position)
                    elif coordinate_type == 'Cartesian':
                        positions.append(any2cartesian(self.lattice, atoms_of_element[j].position)[:3])
                        
                    if ('isContainedConstraints' in kwargs) and kwargs['isContainedConstraints']:
                        constraints.append(self.get_atoms_of_element(self.elements[i].symbol)[j].constraint)
                    if ('isContainedVelocities' in kwargs) and kwargs['isContainedVelocities']:
                        velocities.append(self.get_atoms_of_element(self.elements[i].symbol)[j].velocity)

            elements=np.array(elements)
            numbers=np.array(numbers)
            positions=np.array(positions) 
            constraints=np.array(constraints)
            velocities=np.array(velocities)
     
            poscar={'comment':comment,
                    'lattice':lattice,
                    'elements':elements,
                    'numbers':numbers,
                    'type':coordinate_type,
                    'positions':positions,
                    'constraints':constraints}
            
            if ('isContainedVelocities' in kwargs) and kwargs['isContainedVelocities'] and (velocities != []):
                poscar['velocities']=velocities
                
            return poscar
        
        elif dtype.strip().lower() == 'cif':
            symprec=default_constants.symprec.value
            if 'symprec' in kwargs: symprec=kwargs['symprec']
            angle_tolerance=default_constants.angle_tolerance.value
            if 'angle_tolerance' in kwargs: angle_tolerance=kwargs['angle_tolerance']
            hall_number=default_constants.hall_number.value
            if 'hall_nmber' in kwargs: hall_number=kwargs['hall_number']
            
            structure=self.minimize_clone(isPersist=False)
            sf = StructureFactory(structure).standardize(isPersist=False, symprec=symprec, angle_tolerance=angle_tolerance, hall_number=hall_number)
            structure = sf.structure
            sym = structure._findSymmetryInfo()
            wyckoffs = sym['wyckoffs']
            number = sym['number']
            hall_number = sym['hall_number']
            H_M= sym['hall']
            print(hall_number)
            
            lattice=structure.lattice
            lattice_parameters=structure.lattice_parameters
            #print(structure.sites)
            #print(structure.species)
            #print(structure.atoms)
            #exit()
            
            # positions, elements, wyckoff and multiplicity
            positions=[]
            elements=[]
            wyckoff=[]
            occupancys=[]
            multiplicity=[]
            for i,coord in structure.sites.items():
                positions.append(coord)
                occupancys.append(1)
                elements.append(structure.atoms[i].element.symbol)
                wyckoff.append(wyckoffs[i]['symbol'])
                multiplicity.append(wyckoffs[i]['multiplicity'])

            # formula
            formula=self.composition.formula
            
            # spacegroup
            #if hall != None and format_symbol(H_M) in sgd:
            #    print(1)
            #    equiv=sgd[format_symbol(hm[hall])]
            #else:
            #    print(2)
            #    equiv=sgd[str(number)]
            equiv = SpaceGroup().get_sym_ops(hall_number)
            # volume
            cellvolume=structure.volume

            elements=np.array(elements)
            wyckoff=np.array(wyckoff)
            positions=np.array(positions)
            occupancys=np.array(occupancys)

            cif={'data_':formula,
                  '_cell_length_a':lattice_parameters[0],
                  '_cell_length_b':lattice_parameters[1],
                  '_cell_length_c':lattice_parameters[2],
                  '_cell_angle_alpha':lattice_parameters[3],
                  '_cell_angle_beta':lattice_parameters[4],
                  '_cell_angle_gamma':lattice_parameters[5],
                  '_cell_volume':cellvolume,
                  '_symmetry_space_group_name_H-M':H_M,
                  '_symmetry_Int_Tables_number':number,
                  '_symmetry_equiv_pos_as_xyz':equiv,
                  '_atom_site_symmetry_multiplicity':multiplicity,
                  '_atom_site_label':elements,
                  '_atom_site_wyckoff_symbol':wyckoff,
                  '_atom_site_fract':positions,
                  '_atom_site_occupancy':occupancys}
            return cif
        
        elif dtype.strip().lower() == 'cell':
            numbers=[]
            positions=[]
            for i in range(0, len(self.elements)):
                for j in range(0, len(self.get_atoms_of_element(self.elements[i].symbol))):
                    numbers.append(self.elements[i].z)
                    positions.append(self.get_atoms_of_element(self.elements[i].symbol)[j].position)
            
            numbers=np.array(numbers)
            positions=np.array(positions)
            
            cell={'lattice':self.lattice,
                  'positions':positions,
                  'numbers':numbers}
            return cell
        
        else:
            raise ValueError("unrecognized type in 'dtype'")
#    @profile
    def _findSymmetryInfo(self, symprec=default_constants.symprec.value, angle_tolerance=default_constants.angle_tolerance.value):
        """
        update the information related symmetry.
        
        Arguments: 
            symprec (default=1e-5): precision when to find the symmetry.
            angel_tolerance (default=-1.0): a experimental argument that controls angle tolerance between basis vectors.
        
        Return:
            dictionary array of symmetry information. {'number': dataset['number'],
                                                       'international': dataset['international'],
                                                       'hall': dataset['hall'],
                                                       'hall_number': dataset['hall_number'],
                                                       'pointgroup': dataset['pointgroup'],
                                                       'wyckoffs': wyckoff_set,
                                                       'sites': site_set}
        """
        import collections
        import spglib
        from ..utils.fetch import get_symbol_by_z, get_index_from_positions
        from ..utils.convert import raw2std_position
        
        cell=self.formatting(dtype='cell')
        cell=(cell['lattice'], cell['positions'], cell['numbers'])
        positions=cell[1]
        numbers=cell[2]
        ordered_atoms=[] # ordered atoms' object according to cell
        for i in range(0, len(positions)):
            formated_atom=[get_symbol_by_z(int(cell[2][i])), positions[i][0], positions[i][1], positions[i][2]]
            ordered_atoms.append(self.get_atom(formated_atom))
        dataset=spglib.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=angle_tolerance)
        #print(dataset)
        # wyckoffSite, site
        # note that wyckoffs correspond to the standardize structure by spglib, 
        # while the equivalent_atoms correspond to the raw structure.
        wyckoff=dataset['wyckoffs']
        equivalent_atoms=dataset['equivalent_atoms']
        wyckoff_symbol=[]
        wyckoff_symbol_number=[]
        
        nonequivalent_atoms=collections.OrderedDict() # {iequivalent_atom:[equivalent_atoms]}
        for i in range(0, len(equivalent_atoms)):
            if not(equivalent_atoms[i] in nonequivalent_atoms):
                nonequivalent_atoms[equivalent_atoms[i]]=[i]
            else:
                value=nonequivalent_atoms[equivalent_atoms[i]]
                value.append(i)
                nonequivalent_atoms[equivalent_atoms[i]]=value
        
# =============================================================================
#         std_mapping_to_primitive=dataset['std_mapping_to_primitive']
# =============================================================================
        
        index_of_nonequivalent_atoms=list(nonequivalent_atoms.keys()) # index of nonequivalent atom in atoms.
        for i in index_of_nonequivalent_atoms:
            position=positions[i]
# =============================================================================
#             # Note that, if structure's object is not a primite cell, the output of 'std_mapping_to_primitive' from spglib is not for this structrue,
#             # noly for its primitive cell.
#             position_std=raw2std_position(position, dataset['transformation_matrix'], dataset['origin_shift']) # position in primitive -> position in stander
#             index_std=get_index_from_positions(dataset['std_positions'], position_std)
#             index_primitive=std_mapping_to_primitive[index_std]
#             wyckoff_symbol.append(wyckoff[index_std])
# =============================================================================
            wyckoff_symbol.append(wyckoff[i])
            wyckoff_symbol_number.append(len(nonequivalent_atoms[i]))
        
        site_index=index_of_nonequivalent_atoms #list(nonequivalent_atoms.keys())
        wyckoff_set=collections.OrderedDict() # wyckoff's object set {index:object}
        site_set=collections.OrderedDict() # site's object set {index:object}
        for i in range(0, len(wyckoff_symbol)):
            wyckoffSite={'symbol': wyckoff_symbol[i], 
                         'multiplicity': wyckoff_symbol_number[i], 
                         'position': positions[site_index[i]]}
# =============================================================================
#             wyckoff_set[index_of_nonequivalent_atoms[i]]=wyckoffSite
#             site_set[index_of_nonequivalent_atoms[i]]=positions[site_index[i]]
# =============================================================================
            
            # ensure the atomic indexs of wyckoff_set and sitee_set are consisten with the structure.atoms.
            symbol=get_symbol_by_z(int(cell[2][index_of_nonequivalent_atoms[i]]))
            position=positions[site_index[i]]
            formated_atom=[symbol, position[0], position[1], position[2]]
            atom=self.get_atom(formated_atom=formated_atom)
#            index_in_atoms=get_index_from_collection(atom, self.atoms, entity_type='atom')
            index_in_atoms=self.atoms.index(atom)
            
            wyckoff_set[index_in_atoms]=wyckoffSite
            site_set[index_in_atoms]=positions[site_index[i]]
            
        symmetry_info={'number': dataset['number'],
                       'international': dataset['international'],
                       'hall': dataset['hall'],
                       'hall_number': dataset['hall_number'],
                       'pointgroup': dataset['pointgroup'],
                       'wyckoffs': wyckoff_set,
                       'sites': site_set,
                       'map_of_sites':nonequivalent_atoms}
        return symmetry_info

        

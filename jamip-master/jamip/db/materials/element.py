# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from django.db import models
from ..utils.customField import DictField
from mendeleev import element as mendeleev_element


class Element(models.Model):
    """
    Relationships:
        element
            |-structure
            |-composition
            |-species
            |-atom
            
    Attributes:
        element
            |-symbol: element's symbol. i.e. H
            |-z: atomic number.
            |-name: element's name. i.e. Hydrogen
            |-group: element's group in period table.
            |-period: element's period in period table.
            |-mass: atomic mass.
            |-ionic_radius: ionic radius of element(Unit: pm). 
            |-covalent_radius: covalent radius of element(Unit: pm). 
            |-atomic_radius: atomic radius of element(Unit: pm). 
            |-electronegativity: electronegativity of element.
            |-electron_affinity: electron affinity(Unit: eV). 
            |-ionization_energy: ionization energies(Unit: eV).
            |-electron_configuration: ground state electron configuration.
            # ---------- database ----------
            |-structure_set: collection of structures contained the element.
            |-composition_set: collection of compositions contained the element.
            |-species_set: collection of species contained the element.
            |-atom_set: collection of atoms contained the element.
            # ---------- build-in ----------
            |-structures: collection of structures contained the element.
            |-compositions: collection of compositions contained the element.
            |-species: collection of species contained the element.
            |-atoms: collection of atoms contained the element.
    """
    symbol=models.CharField(primary_key=True, max_length=4)
    z=models.IntegerField(blank=True, null=True)
    name=models.CharField(max_length=20, blank=True, null=True)
    
    group=models.CharField(max_length=20, blank=True, null=True) #models.IntegerField(blank=True, null=True)
    period=models.IntegerField(blank=True, null=True)
    
    mass=models.FloatField(blank=True, null=True)
    # radius unit: pm
    covalent_radius=models.FloatField(blank=True, null=True)
    atomic_radius=models.FloatField(blank=True, null=True)
    vdw_radius=models.FloatField(blank=True, null=True)
    electronegativity=models.FloatField(null=True)

#     ionic_radius=DictField(blank=True, null=True)
#     covalent_radius=DictField(blank=True, null=True)
#     atomic_radius=DictField(blank=True, null=True)
#     vdw_radius=DictField(blank=True, null=True)
#     electronegativity=DictField(blank=True, null=True)
    # unit: eV
    electron_affinity=models.FloatField(blank=True, null=True)
    ionization_energy=DictField(blank=True, null=True)
    electron_configuration=DictField(blank=True, null=True)
    
    # collection of instances
    instances=[]
    @classmethod
    def getInstance(cls, symbol):
        """
        get instance by given symbol.
        
        Arguments:
            symbol: symbol of element. i.e. 'Na'
            
        Return:
            Element's object.
        """
        for instance in Element.instances:
            if symbol == instance.symbol:
                return instance
        else:
            return None
    
    class Meta:
        app_label='materials'
        db_table='element'
        default_related_name='element_set'
        ordering=('z',)
            
    def __str__(self):
        return self.symbol


    _structures=None
    @property
    def structures(self):
        """
        structures contained this element.
        """
        if self._structures is None:
            self._structures=[]
        return self._structures
    
    def _add_structure(self, structure):
        """
        add a structure to this elment.
        
        Arguments:
            structure: structure's object.
            
        Return:
            element's self
        """
        from .structure import Structure
        
        if not isinstance(structure, Structure):
            import warnings
            warnings.warn('not a instance of Structure')
        elif self.structures is None:
            self.structures=[structure]
        else:
            if structure in self.structures:
                import warnings
                warnings.warn('exist in element.structures')
            else:
                self.structures.append(structure)
                
        return self
    
    def _del_structure(self, structure):
        """
        remove a structure to this element.
        
        Arguments:
            structure: structure's object.
            
        Return:
            element's self
        """
        if self.structures is None:
            import warnings
            warnings.warn('element.structures is None')
        else:
            if not structure in self.structures:
                import warnings
                warnings.warn('not exist in element.structures')
            else:
                self.structures.remove(structure)
                
        return self

    _compositions=None
    @property
    def compositions(self):
        """
        compositions contained this element.
        """
        if self._compositions is None:
            self._compositions=[]
        return self._compositions
    
    def _add_composition(self, composition):
        """
        add a composition to this elment.
        
        Arguments:
            composition: composition's object.
            
        Return:
            element's self
        """
        from .composition import Composition
        
        if not isinstance(composition, Composition):
            import warnings
            warnings.warn('not a instance of Composition')
        if self.compositions is None:
            self.compositions=[composition]
        else:
            if composition in self.compositions:
                import warnings
                warnings.warn('exist in element.compositions')
            else:
                self.compositions.append(composition)
                
        return self
    
    def _del_composition(self, composition):
        """
        remove a composition to this element.
        
        Arguments:
            composition: composition's object.
            
        Return:
            element's self
        """
        if self.compositions is None:
            import warnings
            warnings.warn('element.compositions is None')
        else:
            if not composition in self.compositions:
                import warnings
                warnings.warn('not exist in element.compositions')
            else:
                self.compositions.remove(composition)
                
        return self
    
    _species=None
    @property
    def species(self):
        """
        species contained this element.
        """
        if self._species is None:
            self._species=[]
        return self._species
    
    def _add_species(self, species):
        """
        add a species to this elment.
        
        Arguments:
            species: species's object.
            
        Return:
            element's self
        """
        from .species import Species
        
        if not isinstance(species, Species):
            import warnings
            warnings.warn('not a instance of Species')
        if self.species is None:
            self.species=[species]
        else:
            if species in self.species:
                import warnings
                warnings.warn('exist in element.species')
            else:
                self.species.append(species)
                
        return self
    
    def _del_species(self, species):
        """
        remove a species to this element.
        
        Arguments:
            species: species's object.
            
        Return:
            element's self
        """
        if self.species is None:
            import warnings
            warnings.warn('element.species is None')
        else:
            if not species in self.species:
                import warnings
                warnings.warn('not exist in element.species')
            else:
                self.species.remove(species)
                
        return self
    
# =============================================================================
#     The reason we didn't associate with the atom is that the atoms belonging to 
#     the element are distributed in multiple structures. the size of elementl.atoms 
#     will increases fastly with the increaseing of the structures.
# =============================================================================
# =============================================================================
#     _atoms=None
#     @property
#     def atoms(self):
#         """
#         atoms contained this element.
#         """
#         if self._atoms is None:
#             self._atoms=[]
#         return self._atoms
#     
#     def add_atom(self, atom):
#         """
#         add a atom to this element.
#         
#         Arguments:
#             atom: atom's object.
#             
#         Return:
#             element's self
#         """
#         from materials.atom import Atom
#         
#         if not isinstance(atom, Atom):
#             import warnings
#             warnings.warn('not a instance of Atom')
#         if self.atoms is None:
#             self.atoms=[atom]
#         else:
#             if atom in self.atoms:
#                 import warnings
#                 warnings.warn('exist in element.atoms')
#             else:
#                 self.atoms.append(atom)
#                 
#         return self
#     
#     def del_atom(self, atom):
#         """
#         remove a atom to this element.
#         
#         Arguments:
#             atom: atom's object.
#             
#         Return:
#             element's self
#         """
#         if self.atoms is None:
#             import warnings
#             warnings.warn('element.atoms is None')
#         else:
#             if not atom in self.atoms:
#                 import warnings
#                 warnings.warn('not exist in element.atoms')
#             else:
#                 self.atoms.remove(atom)
#                 
#         return self
# =============================================================================
#    @profile
    def create(self, symbol, isPersist=False, **kwargs):
        """
        create a element objects.
        
        Arguments:
            symbol: element's symbol. i.e. 'Na'
            isPersit: if True, save to database. Conversely, only run in memory.
            
            kwargs:
                structures: list of structure's object.
                compositions: list of composition's object or formula. i.e. ['FeSb3', 'CoSb3',...]
                species: list of species's name.
                atoms: list of atom's object.
                
        Returns:
            element's self.
        """
        from ..utils.check import check_symbol
        from ..utils.fetch import get_z_by_symbol, get_name_by_symbol, get_period_by_symbol, get_group_by_symbol, get_mass_by_symbol
        
        # check
        if not check_symbol(symbol): raise ValueError('unknown symbol')
        
        # singleton pattern
        self.symbol=symbol
        instance=Element.getInstance(symbol)
        if not instance is None:
            return instance
# =============================================================================
#         instances=Element.instances
#         if len(instances) > 1:
#             for instance in instances[:-1]:
#                 if symbol == instance.symbol:
#                     return instance
# =============================================================================
        
        self.z=get_z_by_symbol(symbol)
        self.name=get_name_by_symbol(symbol)
        self.period=get_period_by_symbol(symbol)
        self.group=get_group_by_symbol(symbol)
        self.mass=get_mass_by_symbol(symbol) 

        if isPersist:
            self.save()
            
        Element.instances.append(self)
        return self

    def supplement(self):
        from mendeleev import element
        melm = element(self.symbol)
        self.covalent_radius = melm.covalent_radius_cordero
        if self.covalent_radius != None:
            self.covalent_radius = round(self.covalent_radius,6)
        self.atomic_radius = melm.atomic_radius
        self.vdw_radius = melm.vdw_radius
        if self.vdw_radius != None:
            self.vdw_radius = round(self.vdw_radius,6)
        self.electronegativity = melm.electronegativity('pauling')
        self.electron_affinity = melm.electron_affinity
        self.mass = melm.mass
        #ie = melm.ionenergies
        #ir0 = getIonicRadius(melm,2,'VI')
        return self

        

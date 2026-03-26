# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from django.db import models


class Species(models.Model):
    """
    species.
    
    Relationships:
        species
            |- structure
            |- element
            |- atom
            
    Attributes:
        species
            |- element
            |- name
            |- ox
            # ---------- database ----------
            |- structure_set
            |- atom_set
            # ---------- build-in ----------
            |- structures
            |- atoms
    """
    # relationship 
    element=models.ForeignKey('Element', blank=True, null=True, on_delete=models.PROTECT)
    name=models.CharField(max_length=8, primary_key=True) # i.e. Na+
    ox=models.FloatField(blank=True, null=True) # oxidation state
    
    # collection of instances
    instances=[]
    @classmethod
    def getInstance(cls, name):
        """
        get instance by given name.
        
    Arguments:
        name: name of species. i.e. Fe2+
        
    Return:
        Species's object.
        """
        for instance in Species.instances:
            if name == instance.name:
                return instance
        else:
            return None
    
    class Meta:
        app_label='materials'
        db_table='species'
        default_related_name='species_set'
    
    def __str__(self):
        return self.name
    
    _structures=None
    @property
    def structures(self):
        """
        structures contained this species.
        """
        if self._structures is None:
            self._structures=[]
        return self._structures
    
    def _add_structure(self, structure):
        """
        add a structure to this species.
        
        Arguments:
            structure: structure's object.
            
        Return:
            species's self
        """
        from .structure import Structure
        
        if not isinstance(structure, Structure):
            import warnings
            warnings.warn('not a instance of Structure')
        if self.structures is None:
            self.structures=[structure]
        else:
            if structure in self.structures:
                import warnings
                warnings.warn('exist in species.structure')
            else:
                self.structures.append(structure)
                
        return self
    
    def _del_structure(self, structure):
        """
        remove a structure to this species.
        
        Arguments:
            structure: structure's object.
            
        Return:
            species's self
        """
        if self.structures is None:
            import warnings
            warnings.warn('species.structure is None')
        else:
            if not structure in self.structures:
                import warnings
                warnings.warn('not exist in species.structure')
            else:
                self.structures.remove(structure)
                
        return self
    
# =============================================================================
#     _atoms=None
#     @property
#     def atoms(self):
#         """
#         atoms contained this species.
#         """
#         if self._atoms is None:
#             self._atoms=[]
#         return self._atoms
#     
#     def add_atom(self, atom):
#         """
#         add a atom to this species.
#         
#         Arguments:
#             atom: atom's object.
#             
#         Return:
#             species's self
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
#                 warnings.warn('exist in species.atoms')
#             else:
#                 self.atoms.append(atom)
#                 
#         return self
#     
#     def del_atom(self, atom):
#         """
#         remove a atom to this species.
#         
#         Arguments:
#             atom: atom's object.
#             
#         Return:
#             species's self
#         """
#         if self.atoms is None:
#             import warnings
#             warnings.warn('species.atoms is None')
#         else:
#             if not atom in self.atoms:
#                 import warnings
#                 warnings.warn('not exist in species.atoms')
#             else:
#                 self.atoms.remove(atom)
#                 
#         return self
# =============================================================================
    
    def create(self, name, isPersist=False, **kwargs):
        """
        create species's object. Note that oxidation state (ox) of this species is extracted from the variable 'name'.
        
        Arguments:
            name: name of species. i.e. Fe2+
            isPersist: if True, save to database. Conversely, only run in memory.
            
            kwargs:
# =============================================================================
#                 structures: collection of structure's object.
#                 atoms: collection of atom's object.
# =============================================================================
                
        Returns:
            species's object.
        """
        from .element import Element
        from ..utils.check import check_species
        from ..utils.fetch import get_symbol_from_species, get_oxidation_state_from_species
        from ..utils.convert import formatting_species
        
        if check_species(name):
            # singleton pattern
            self.name=formatting_species(name)
            instance=Species.getInstance(name)
            if not instance is None:
                return instance
            
            symbol=get_symbol_from_species(name)
            ox=get_oxidation_state_from_species(name)
            
            self.ox=ox
            self.element=Element().create(symbol=symbol, isPersist=isPersist)
        else:
            raise ValueError('unknown name')
        
        if isPersist:
            self.save()
            
        Species.instances.append(self)
        return self
        

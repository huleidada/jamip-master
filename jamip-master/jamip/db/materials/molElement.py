# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from django.db import models
from ..utils.customField import DictField

class MolElement(models.Model):
    """
    molecular element。
    
    Relationships:
        element
            |- structure
            |- composition
            |- atom
            
    Attrubitues:
        element
            |- symbol
            |- z
            |- name
            |- group
            |- period
            |- mass
            |- electronegativity
            # ---------- database ----------
            |- structure_set: collection of structures contained the element.
            |- composition_set: collection of compositions contained the element.
            |- atom_set: collection of atoms contained the element.
            # ---------- build-in ----------
            |- structures: collection of structures contained the element.
            |- compositions: collection of compositions contained the element.
            |- atoms: collection of atoms contained the element.
    """
    symbol=models.CharField(primary_key=True, max_length=4)
    z=models.IntegerField()
    name=models.CharField(max_length=20)
    
    group=models.IntegerField()
    period=models.IntegerField()
    
    mass=models.FloatField(blank=True, null=True)
    # radius unit: pm
    ionic_radius=DictField(null=True)
    covalent_radius=DictField(null=True)
    atomic_radius=DictField(null=True)
    vdw_radius=DictField(null=True)
    
#     electronegativity=models.FloatField(blank=True, null=True)
    electronegativity=DictField(null=True)
    # unit: eV
    electron_affinity=models.FloatField(null=True)
    ionization_energy=DictField(null=True)
    
    electron_configuration=DictField(null=True)
    
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
        for instance in MolElement.instances:
            if symbol == instance.symbol:
                return instance
        else:
            return None
    
    class Meta:
        app_label='materials'
        db_table='molElement'
        default_related_name='element_set'
            
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
        from .molStructure import MolStructure
        
        if not isinstance(structure, MolStructure):
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
        from .molComposition import MolComposition
        
        if not isinstance(composition, MolComposition):
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
        instance=MolElement.getInstance(symbol)
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
        self.mass=get_mass_by_symbol(symbol) if not(self.mass == '*') else None

        if isPersist:
            self.save()
            
        MolElement.instances.append(self)
        return self
    

# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from django.db import models
from ..utils.customField import ListField, NumpyArrayField
from ..utils.variables import default_constants


class Atom(models.Model, object):
    """
    atom.
    
    Relationships:
        atom
            |- structure
            |- element
            |- species
            
    Attributes:
        atom
            |- structure
            |- site
            |- element
            |- species
#            |- ox
            |- position
            |- force
            |- velocity
            |- constraint
            |- magmom
            |- charge
            |- volume
            |- occupancy
    """
    
# =============================================================================
#     structure=models.ForeignKey('Structure', null=True, on_delete=models.PROTECT)
#     element=models.ForeignKey('Element', null=True, on_delete=models.PROTECT)
# =============================================================================
    structure=models.ForeignKey('Structure', null=True, on_delete=models.PROTECT)
    element=models.ForeignKey('Element', null=True, on_delete=models.PROTECT)
    species=models.ForeignKey('Species', null=True, on_delete=models.PROTECT)
    
#    ox=models.FloatField(blank=True, null=True)
    
    position=ListField(blank=True, null=True) # [3, 3] (float)
    force=NumpyArrayField(blank=True, null=True) # [3, 3] (float)
    velocity=NumpyArrayField(blank=True, null=True) # [3, 3] (float)
    
    constraint=NumpyArrayField(blank=True, null=True) # [3, 3] (boolean)
    
    magmom=models.FloatField(blank=True, null=True)
    charge=models.FloatField(blank=True, null=True) # effective charge
    volume=models.FloatField(blank=True, null=True) # atomic volume
    
    # symmetry
    occupancy=models.FloatField(default=1) # value is from 0 to 1.
    
    class Meta:
        app_label='materials'
        db_table='atom'
        default_related_name='atom_set'
        
# =============================================================================
#      def __str__(self):
#          if self.position is None:
#              return '%s: [None, None, None]' %self.element.symbol
#          else:
#              return '%s [%f, %f, %f]' %(self.element.symbol, self.position[0], self.position[1], self.position[2])
# =============================================================================
    def __str__(self):
        return '[{}, {}, {}, {}]'.format(self.element.symbol, self.position[0], self.position[1], self.position[2])
    
#    @profile
    def create(self, formated_atom, isNormalizingCoordinate=default_constants.isNormalizingCoordinate.value, isPersist=False, **kwargs):
        """
        create a atom object.
        
        Arguments:
# =============================================================================
#             element: element's symbol of this atom. i.e. Na
#             position: atomic position.
# =============================================================================
            formated_atom (default='Direct'): formated atom. The valid formation:
                ['Na', 0.1, 0.0, 0.0, 'Direct']
                ['Na', 0.1, 0.0, 0.0]
                ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                
                contain species information:
                ['Na1+', 0.1, 0.0, 0.0, 'Direct']
                ['Na1+', 0.1, 0.0, 0.0]
                ['Na1+', 5.234, 0.0, 0.0, 'Cartesian']
                
            isNormalizingCoordinate (default=True): whether to remove the periodic boundary condition, 
                ensure the value of atomic coordinate is between 0 and 1 (i.e. 1.3 -> 0.3).
            isPersit: if True, save to database. Conversely, only run in memory.
            
            kwargs:
                lattice: lattice parameter of structure. i.e.
                    [[x1,y1,z1],
                     [x2,y2,z2],
                     [x3,y3,z3]]
# =============================================================================
#                     
#                 structure:
#                 site:
#                 
#                 species: species's name. i.e. Fe2+
#                 ox:
#                 velocity:
#                 force:
#                 constraint:
#                 magmom:
#                 charge:
#                 volume:
#                 occupancy:
#                 wyckoffSite:
# =============================================================================
                
        Returns:
            atom's object.
        """
        import numpy as np
        from ..utils.check import check_formated_atom_only_direct, check_formated_atom_only_cartesian, check_symbol, check_species
        from ..utils.fetch import get_symbol_from_species
        from ..utils.convert import cartesian2direct, normalize_position
        from .element import Element
        from .species import Species
        
        symbol=None
        position=None
        
        if check_formated_atom_only_direct(formated_atom):
            symbol=formated_atom[0] if check_symbol(formated_atom[0]) else get_symbol_from_species(formated_atom[0])
            position=np.array(formated_atom[1:4])
        elif check_formated_atom_only_cartesian(formated_atom):
            lattice=None
            if 'lattice' in kwargs:
                lattice=kwargs['lattice']
                
            symbol=formated_atom[0] if check_symbol(formated_atom[0]) else get_symbol_from_species(formated_atom[0])
            position=cartesian2direct(lattice=lattice,position=formated_atom[1:4])
        else:
            raise ValueError('unknown formated_atom')
        
        self.element=Element().create(symbol=symbol, isPersist=isPersist)
        self.position=normalize_position(position=position, dtype='Direct')[:3] if isNormalizingCoordinate else position
        
        # species
        if check_species(formated_atom[0]):
            self.species=Species().create(name=formated_atom[0])
            
        if isPersist:
            self.save()
            
        return self
    
    def to_formated_atom(self, dtype='Direct', isContainedSpecies=True):
        """
        get the formated atom.
            ['Na', 0.1, 0.0, 0.0, 'Direct']
            ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                    
            contain species information:
            ['Na1+', 0.1, 0.0, 0.0, 'Direct']
            ['Na1+', 5.234, 0.0, 0.0, 'Cartesian']
            
        Arguments:
            dtype (defalut='Direct'): type of atomic position. 'Direct' or 'Cartesian'.
            isContainedSpecies (default=True): whether to contain the species information.
        """
        formated_atom=None
        if dtype.lower().startswith('d'): # 'Direct'
            position=self.position
            formated_atom=[self.element.symbol, position[0], position[1], position[2], 'Direct']
        elif dtype.lower().startswith('c'): # 'Cartesian'
            from ..utils.convert import direct2cartesian
            
            position=self.position
            position=direct2cartesian(lattice=self.structure.lattice, position=position)
            
            formated_atom=[self.element.symbol, position[0], position[1], position[2], 'Cartesian']
        else:
            import warnings
            warnings.warn('unknown dtpye')
            
        if isContainedSpecies and not(self.species is None):
            formated_atom[0]=self.species.name
            
        return formated_atom
    
    
    

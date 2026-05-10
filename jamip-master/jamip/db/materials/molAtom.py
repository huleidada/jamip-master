# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from django.db import models

from ..utils.customField import NumpyArrayField

class MolAtomError(Exception):
    pass

class MolAtom(models.Model):
    """
    Molecular Atom.
    
    Relationships:
        atom
            |- structure
            |- element
            
    Attributes:
        atom
            |- structure
            |- element
            |- position
            |- magmom
            |- charge
            |- volume
#            |- ox
    """
    # relationship
    structure=models.ForeignKey('MolStructure', null=True, on_delete=models.PROTECT)
    element=models.ForeignKey('MolElement', null=True, on_delete=models.PROTECT)
    
    position=NumpyArrayField(blank=True, null=True)
    
    magmom=models.FloatField(blank=True, null=True)
    charge=models.FloatField(blank=True, null=True) # effective charge
    volume=models.FloatField(blank=True, null=True) # atomic volume
    
#    ox=models.FloatField(blank=True, null=True) # oxidation state
    
    class Meta:
        app_label='materials'
        db_table='molAtom'
        default_related_name='atom_set'
    
# =============================================================================
#     def __str__(self):
#         if self.position is None:
#             return '%s: [None, None, None]' %self.element.symbol
#         else:
#             return '%s [%f, %f, %f]' %(self.element.symbol, self.position[0], self.position[1], self.position[2])
# =============================================================================
    def __str__(self):
        return '[{}, {}, {}, {}]'.format(self.element.symbol, self.position[0], self.position[1], self.position[2])
    
    def create(self, formated_atom, isPersist=False, **kwargs):
        """
        create a atom object.
        
        Arguments:
# =============================================================================
#             element: element's symbol of this atom. i.e. Na
#             position: atomic position.
# =============================================================================
            formated_atom: formated atom. The valid formation:
                ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                
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
        from ..utils.check import check_formated_atom_only_cartesian, check_symbol
        from .molElement import MolElement
        
        symbol=None
        position=None
        
        if check_formated_atom_only_cartesian(formated_atom):
            symbol=formated_atom[0]
            position=np.array(formated_atom[1:4])
        else:
            raise ValueError('unknown formated_atom')
            
        self.element=MolElement().create(symbol=symbol, isPersist=isPersist)
        self.position=position
            
        if isPersist:
            self.save()
            
        return self
    
    def to_formated_atom(self, *args, **kwargs):
        """
        get the formated atom.
            ['Na', 5.234, 0.0, 0.0, 'Cartesian']
        """
        return [self.element.symbol, self.position[0], self.position[1], self.position[2], 'Cartesian']

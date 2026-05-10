# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from django.db import models
from ..utils.customField import NumpyArrayField,DictField


class Spacegroup(models.Model):
    """
    spacegroup.
    
    Relationships:
        spacegroup
            |- structure
            
    Attributes:
        spacegroup
            |- number
            |- international
            |- hm
            |- hall
            |- pointgroup
            |- pearson
            |- schoenflies
            |- lattice_system
    """
    # relationship
    
    number=models.IntegerField(primary_key=True)
    international=models.CharField(max_length=20, blank=True, null=True)
    #hm=models.CharField(max_length=20, blank=True, null=True) # hall number
    hm=models.IntegerField(blank=True, null=True) # hall number
    hall=models.CharField(max_length=20, blank=True, null=True)
    #pointgroup=models.CharField(max_length=20, blank=True, null=True)
    pearson=models.CharField(max_length=20, blank=True, null=True)
    schoenflies=models.CharField(max_length=20, blank=True, null=True)
    lattice_system=models.CharField(max_length=20, blank=True, null=True)
    centerosymmetric=models.BooleanField(blank=True, null=True)
    
    # collection of instances
    instances=[]
    @classmethod
    def getInstance(cls, number):
        """
        get instance by given number.
        
        Arguments:
            number: number of spacegroup.
            
        Return:
            Spacegroup's object.
        """
        for instance in Spacegroup.instances:
            if number == instance.number:
                return instance
        else:
            return None        
    
    class Meta:
        app_label='materials'
        db_table='spacegroup'
        default_related_name='spacegroup_set'
    
    def __str__(self):
        return '%s (%d)' %(self.international, self.number)
    
    _structures=None
    @property
    def structures(self):
        """
        structures contained this spacegroup.
        """
        if self._structures is None:
            self._structures=[]
        return self._structures
    
    def _add_structure(self, structure):
        """
        add a structure to this spacegroup.
        
        Arguments:
            structure: structure's object.
            
        Return:
            spacegroup's self
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
                warnings.warn('exist in spacegroup.structure')
            else:
                self.structures.append(structure)
                
        return self
    
    def _del_structure(self, structure):
        """
        remove a structure to this spacegroup.
        
        Arguments:
            structure: structure's object.
            
        Return:
            spacegroup's self
        """
        if self.structures is None:
            import warnings
            warnings.warn('spacegroup.structure is None')
        else:
            if not structure in self.structures:
                import warnings
                warnings.warn('not exist in spacegroup.structure')
            else:
                self.structures.remove(structure)
                
        return self
    
    def create(self, number, isPersist=False, **kwargs):
        """
        create spacegroup's object.
        
        Arguments:
            number: number of spacegroup.
            isPersist: if True, save to database. Conversely, only run in memory.
            
            kwargs:
                operations: collection of operation's object.
                wyckoffSites: collection of wyckoffSite's object.
                structures: collection of structure's object.
                
                international: international short symbol.
                hm: hall number.
                hall: hall symbol.
                pearson: 
                schoenflies:
                lattice_system:
                centerosymmetric:
                
        Returns:
            spacegroup's object.
        """
        # singleton pattern
        self.number=number
        instance=Spacegroup.getInstance(number)
        if not instance is None:
            return instance
        
        if 'international' in kwargs:
            self.international=kwargs['international']
        if 'hm' in kwargs:
            self.hm=kwargs['hm']
        if 'hall' in kwargs:
            self.hall=kwargs['hall']
        if 'pearson' in kwargs:
            self.pearson=kwargs['pearson']
        if 'schoenflies' in kwargs:
            self.schoenflies=kwargs['schoenflies']
        if 'lattice_system' in kwargs:
            self.lattice_system=kwargs['lattice_system']
        if 'centerosymmetric' in kwargs:
            self.centerosymmetric=kwargs['centerosymmetric']
        
        if isPersist:
            self.save()
            
        Spacegroup.instances.append(self)
        return self

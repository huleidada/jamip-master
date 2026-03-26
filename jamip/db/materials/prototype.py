# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from django.db import models


class Prototype(models.Model):
    """
    prototype.
    
    Relationships:
        prototype
            |-structure
            |-composition
                
    Attributes:
        prototype
            |-composition: composition of prototype.
            |-structure_of_prototype: structure of prototype.
            |-name: name of prototype.
            # ---------- database ----------
            |-structure_set: collection of structures belong to the prototype.
            # ---------- build-in ----------
            |-structures: collection of structures belong to the prototype.
    """
    # relationships
    composition=models.ForeignKey('Composition', blank=True, null=True, on_delete=models.PROTECT)
    structure_of_prototype=models.ForeignKey('Structure', blank=True, null=True, on_delete=models.PROTECT, related_name='+')
    #structure_of_prototype=models.ForeignKey('Structure', blank=True, null=True)
    
    name=models.CharField(max_length=80, primary_key=True)
    
    # collection of instances
    instances=[]
    @classmethod
    def getInstance(cls, name):
        """
        get instance by given name.
        
        Arguments:
            name: name of prototype. i.e. xxx
            
        Return:
            prototype's object.
        """
        for instance in Prototype.instances:
            if name == instance.name:
                return instance
        else:
            return None
    
    class Meta:
        app_label='materials'
        db_table='prototype'
        #default_related_name='prototype_set'
        
    def __str__(self):
        return self.name
    
    _structures=None
    @property
    def structures(self):
        """
        structures contained this prototype.
        """
        if self._structures is None:
            self._structures=[]
        return self._structures
    
    def _add_structure(self, structure):
        """
        add a structure to this prototype.
        
        Arguments:
            structure: structure's object.
            
        Return:
            prototype's self
        """
        if self.structures is None:
            self.structures=[structure]
        else:
            if structure in self.structures:
                import warnings
                warnings.warn('exist in prototype.structure')
            else:
                self.structures.append(structure)
                
        return self
    
    def _del_structure(self, structure):
        """
        remove a structure to this prototype.
        
        Arguments:
            structure: structure's object.
            
        Return:
            prototype's self
        """
        if self.structures is None:
            import warnings
            warnings.warn('prototype.structure is None')
        else:
            if not structure in self.structures:
                import warnings
                warnings.warn('not exist in prototype.structure')
            else:
                self.structures.remove(structure)
                
        return self
    
    def create(self, name, structure_of_prototype, isPersist=False, **kwargs):
        """
        create prototype's object.
        
        Arguments:
            name: name of prototype.
            structure_of_prototype: structure of prototype.
            isPersist (default=False): whether to save to the database.
            
            kwargs:
#                formula_of_composition: composition's formula  of prototype.
                
#                structures: collection of structure's object, which belong to this prototype.
        
        Returns:
            prototype's object.
        """
        from materials.structure import Structure
        from materials.composition import Composition
        
        self.name=name
        if isinstance(structure_of_prototype, Structure):
            self.structure_of_prototype=structure_of_prototype
        else:
            raise ValueError("structure_of_prototype is not a structure's object")

        if 'formula_of_composition' in kwargs:
            formula_of_composition = kwargs['formula_of_composition']
            if isinstance(formula_of_composition, Composition):
                self.composition=formula_of_composition
            elif isinstance(formula_of_composition, Composition):
                self.composition=Composition().create(formula_of_composition)
        else:
            self.composition=self.structure_of_prototype.composition
        
        if isPersist:
            self.save()
        
        Prototype.instances.append(self)
        return self
    
    

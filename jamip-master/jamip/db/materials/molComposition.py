# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from django.db import models

class MolComposition(models.Model):
    """
    Molecular Composition
    
    Relationships:
        composition
            |- structure
            |- element
        
    Attributes:
        composition
            |- formula: normalized composition. i.e. PbTiO3
            |- generic: generalized composition. i.e. ABO3
            |- mass: mass per formula.
            # ---------- database ----------
            |- structure_set
            |- element_set
            # ---------- build-in ----------
            |- structures
            |- elements
    """
    # relationship
    element_set=models.ManyToManyField('MolElement')
    
    formula=models.CharField(primary_key=True, max_length=255)
    generic=models.CharField(max_length=255, blank=True, null=True)
    mass=models.FloatField(null=True)
    
    # collection of instances
    instances=[]
    @classmethod
    def getInstance(cls, formula):
        """
        get instance by given formula.
        
        Arguments:
            formula: normalized composition. i.e. PbTiO3
            
        Return:
            Composition's object.
        """
        for instance in MolComposition.instances:
            if formula == instance.formula:
                return instance
        else:
            return None
    
    class Meta:
        app_label='materials'
        db_table='molComposition'
        default_related_name='composition_set'
        
    def __str__(self):
        return self.formula
    
    _structures=None
    @property
    def structures(self):
        """
        structures contained this composition.
        """
        if self._structures is None:
            self._structures=[]
        return self._structures
    
    def _add_structure(self, structure):
        """
        add a structure to this composition.
        
        Arguments:
            structure: structure's object.
            
        Return:
            composition's self
        """
        from .molStructure import MolStructure
        
        if not isinstance(structure, MolStructure):
            import warnings
            warnings.warn('not a instance of Structure')
        if self.structures is None:
            self.structures=[structure]
        else:
            if structure in self.structures:
                import warnings
                warnings.warn('exist in composition.structure')
            else:
                self.structures.append(structure)
                
        return self
    
    def _del_structure(self, structure):
        """
        remove a structure to this composition.
        
        Arguments:
            structure: structure's object.
            
        Return:
            composition's self
        """
        if self.structures is None:
            import warnings
            warnings.warn('composition.structure is None')
        else:
            if not structure in self.structures:
                import warnings
                warnings.warn('not exist in composition.structure')
            else:
                self.structures.remove(structure)
                
        return self
    
    _elements=None
    @property
    def elements(self):
        """
        elements contained this composition.
        """
        if self._elements is None:
            self._elements=[]
        return self._elements
    
    def _add_element(self, element):
        """
        add a element to this composition.
        
        Arguments:
            element: element's object.
            
        Return:
            composition's self
        """
        from .molElement import MolElement
        
        if not isinstance(element, MolElement):
            import warnings
            warnings.warn('not a instance of Element')
        elif self.elements is None:
            self.elements=[element]
        else:
            if element in self.elements:
                import warnings
                warnings.warn('exist in composition.element')
            else:
                self.elements.append(element)
                
        return self
    
    def _del_element(self, element):
        """
        remove a element to this composition.
        
        Arguments:
            element: element's object.
            
        Return:
            composition's self
        """
        if self.elements is None:
            import warnings
            warnings.warn('composition.element is None')
        else:
            if not element in self.elements:
                import warnings
                warnings.warn('not exist in composition.element')
            else:
                self.elements.remove(element)
                
        return self

    _mass=None
    @property
    def mass(self):
        """
        molecular mass of this composition.
        """
        import re
        from .molElement import MolElement
     
        element_pat=re.compile('([A-z][a-z]?)(\d*)')
        elements=[]
        for (symbol, number) in element_pat.findall(self.formula):
            if number == '':
                number=1
            else:
                number=int(number)
            elements.extend([symbol]*number)
     
        m0=0.0 # mass of formula
        for e0 in elements:
            m0 += MolElement().create(symbol=e0).mass
        self._mass=m0
        return self._mass
    
    def create(self, formula, isPersist=False, **kwargs):
        """
        create a composition object.
        
        Arguments:
            formula: normalized composition. i.e. Pb1TiO3
                Note that need to be also given when atomic number of element is 1.
            isPersit: if True, save to database. Conversely, only run in memory.
            
            kwargs:
                prototypes: collection of prototype's object.
                structures: collection of structure's object.
                elements: collection of element's object or symbol.
                generic: generalized composition. i.e. ABO3
        
        Returns:
            composition's object.
        """
        # singleton pattern
        self.formula=formula
        instance=MolComposition.getInstance(formula)
        if not instance is None:
            return instance
        
        if 'generic' in kwargs:
            generic=kwargs['generic']
            self.generic=generic
        
        if isPersist:
            self.save()
            
        MolComposition.instances.append(self)
        return self

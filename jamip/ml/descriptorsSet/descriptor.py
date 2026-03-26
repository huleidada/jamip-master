# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod


class Descriptor:
    __metaclass__=ABCMeta
    
    def __init__(self, structures):
        self._structures=structures
        self._feature=None        
        self._builder=None
    
    def set_structures(self, structures):    
        self._structures=None
        for structure in list(structures):
            self.add_structure(structure)
        
        return self
    
    def add_structure(self, structure, isUpdatedInfo=False):
        if self._structures is None:
            self._structures=[]
        if not structure in self._structures:
            self._structures.append(structure)
        else:
            import warnings
            warnings.warn('this descriptor has existed in structures')
        
        # update
        if isUpdatedInfo: self.set_feature(descriptor=self.feature.columns[0])
        
        return self
    
    def remove_structure(self, structure, isUpdatedInfo=False):
        import warnings
        
        if self._structures is None:
            warnings.warn('structures is None')
        elif not structure in self._structures:
            warnings.warn('not exist in structures')
        else:
            self._structures.remove(structure)
        
        # update
        if isUpdatedInfo: self.set_feature(descriptor=self.feature.columns[0])
        
        return self
        
    @property
    def feature(self):
        return self._feature
    
    def set_feature(self, descriptor):
        self._feature=self._getValues(descriptor)
        return self
    
    @abstractmethod
    def _getValues(self, descriptor):
        raise NotImplementedError()
        
    @property
    def builder(self):
        return self._builder
    

class DescriptorBuilder:
    
    def __init__(self, structures):
        self.descriptor=Descriptor(structures=structures)
        self.descriptor._builder=self
        
    def get_result(self):
        return self.descriptor
    
    def set_feature(self, descriptor):
        self.descriptor.set_feature(descriptor)
        return self
    
    def set_structures(self, structures):
        self.descriptor.set_structures(structures)
        return self
    
    def add_structure(self, structure, isUpdatedInfo=False):
        self.descriptor.add_structure(structure=structure, isUpdatedInfo=isUpdatedInfo)
        return self
    
    def remove_structure(self, structure, isUpdatedInfo=False):
        self.descriptor.remove_structure(structure=structure, isUpdatedInfo=isUpdatedInfo)
        return self

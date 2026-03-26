# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import pandas as pd
from abc import ABCMeta, abstractmethod
from .property.propertyDescriptor import PropertyDescriptorBuilder
from .structure.structureDescriptor import StructureDescriptorBuilder


class DescriptorsSet:
    __metaclass__=ABCMeta
    __methods__=[]
    
    def __init__(self, structures):
        self._structures=structures
        self._features=None        
        self._builder=None
    
    @property
    def features(self):
        return self._features
    
    def set_features(self, descriptors):        
        for descriptor in list(descriptors):
            self.add_feature(descriptor)
        
        return self
    
    def add_feature(self, descriptor):
        if self._features is None:
            self._features=pd.DataFrame()
        if not descriptor in self._features.columns:
            self._features[descriptor]=self._getValues(descriptor)
        else:
            import warnings
            warnings.warn('this descriptor has existed in features')
        
        return self
    
    def remove_feature(self, descriptor):
        import warnings
        
        if self._features is None:
            warnings.warn('features is None')
        elif not descriptor in self._features.columns:
            warnings.warn('not exist in features')
        else:
            self._features.pop(descriptor)
        
        return self

    def _get_feature_transformation(self, expression_of_descriptor):
        """
        for example:
            
            type-1: A^2, A^3, ..., A^x, A+A^2, tanh(A), etc.
            type-2: A*B, A/B, A+B, etc.

        Parameters
        ----------
        expression_of_descriptor : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        descriptors=self.features.columns
        
        variates={}
        for d0 in descriptors:
            if d0 in expression_of_descriptor: variates[d0]=self.features[d0]

        if variates != {}: 
            self._features[expression_of_descriptor]=eval(expression_of_descriptor, variates)
            return self.features[expression_of_descriptor]
        else:
            import warnings
            warnings.warn("{} doesn't exist in self.features".format(expression_of_descriptor))
            return None
        
    
    def feature_transformations(self, expression_of_descriptors):
        """
        for example:
            
            type-1: A^2, A^3, ..., A^x, A+A^2, tanh(A), etc.
            type-2: A*B, A/B, A+B, etc.

        Parameters
        ----------
        expression_of_descriptors : TYPE
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
                
        for eod in expression_of_descriptors:
            self._get_feature_transformation(expression_of_descriptor=eod)
            
        return self

    def add_structures(self,structures):
        if self._features is None:
            self._structures.extend(structures)
        elif isinstance(self._features,pd.DataFrame):
            pddata = self._features
            descriptors = pddata.columns
            # clear current data %
            self._structures = structures
            self._features=pd.DataFrame()
            self.set_features(descriptors)
            # merge data %
            self._features = pd.concat([pddata,self._features])
        else:
            raise TypeError('error in features data type!')

        return self

    def add_methods(self,methods):
        # add external methods %
        from types import MethodType

        for method in methods:
            self.__methods__.append(method.__name__)
            self.__setattr__(method.__name__,MethodType(method,self))
        return self
    
    @abstractmethod
    def _getValues(self, descriptor):
        # raise NotImplementedError
        from ml.utils.check import isPropertyDescriptor, isStructureDescriptor
       
        if descriptor in self.__methods__:
            return getattr(self,descriptor)()
        elif isPropertyDescriptor(descriptor):
            return PropertyDescriptorBuilder(structures=self._structures).set_feature(descriptor).get_result().feature
        elif isStructureDescriptor(descriptor):
            return StructureDescriptorBuilder(structures=self._structures).set_feature(descriptor).get_result().feature
        else:
            msg = 'unknown descriptor %s' %descriptor
            raise ValueError(msg)
            
    @property
    def builder(self):
        return self._builder

class DescriptorsSetBuilder:
    
    def __init__(self, structures):

        self.descriptorsSet=DescriptorsSet(structures=structures)
        self.descriptorsSet._builder=self
        
    def get_result(self):
        return self.descriptorsSet
    
    def set_features(self, descriptors):
        self.descriptorsSet.set_features(descriptors)
        return self
    
    def add_structures(self, structures):
        self.descriptorsSet.add_structures(structures)
        return self

    def add_methods(self,methods):
        self.descriptorsSet.add_methods(methods)

    def save(self,filename):
        import warnings
        import numpy as np
        from copy import deepcopy
        df = deepcopy(self.descriptorsSet._features)
        for feature in df.columns:
            if df[feature].dtype == object: 
                types = [type(i) for i in df[feature]]
                if str in types: continue
                elif np.ndarray in types or list in types:
                    np.savez(feature,*df[feature])
                    df[feature] = np.nan
                else:
                    msg = 'Data types may change after save.'
                    warnings.warn(msg)
        df.to_csv(filename,index=False)

    @classmethod
    def load(self, filename):
        from os.path import exists
        import numpy as np
        df = pd.read_csv(filename) 
        for feature in df.columns:
            if exists(feature+'.npz'):
                with np.load(feature+'.npz') as data:
                    df[feature]=[data[file] for file in data]
        return df




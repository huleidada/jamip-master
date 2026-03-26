# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from ..descriptor import Descriptor, DescriptorBuilder
from functools import lru_cache
@lru_cache()
def get_element(elm):
    from jump2.db.connect import Element
    queryset = Element.objects.filter(symbol=elm.symbol)
    if len(queryset) != 1:
        mag = 'invalid input element %s' %elm.symbol
        raise ValueError(msg)
    return queryset[0]

    

class PropertyDescriptor(Descriptor):
    
    def __init__(self, structures):
        super().__init__(structures=structures)
        self._elements=None

    def _getValues(self, descriptor):
        import warnings
        warnings.warn('need to implement codes')
        if descriptor == 'volume':
            return [s.volume for s in self._structures] 
        elif descriptor == 'natoms':
            return [s.natoms for s in self._structures] 
        elif descriptor == 'lattice':
            return [s.lattice for s in self._structures] 
        elif descriptor == 'a':
            return [s.lattice_parameters[0] for s in self._structures] 
        elif descriptor == 'b':
            return [s.lattice_parameters[1] for s in self._structures] 
        elif descriptor == 'c':
            return [s.lattice_parameters[2] for s in self._structures] 
        elif descriptor == 'alpha':
            return [s.lattice_parameters[3] for s in self._structures] 
        elif descriptor == 'beta':
            return [s.lattice_parameters[4] for s in self._structures] 
        elif descriptor == 'gamma':
            return [s.lattice_parameters[5] for s in self._structures] 

        elif descriptor == 'spacegroup':
            return [s.spacegroup.number for s in self._structures]
        elif descriptor == 'composition':
            return [s.composition.formula for s in self._structures]
        elif descriptor == 'symbol':
            _map = lambda elms: [e.symbol for e in elms]
            return [_map(s.elements) for s in self._structures]
        elif descriptor == 'z': 
            _map = lambda elms: [e.z for e in elms]
            return [_map(s.elements) for s in self._structures]
        elif descriptor == 'group': 
            _map = lambda elms: [e.group for e in elms]
            return [_map(s.elements) for s in self._structures]
        elif descriptor == 'period': 
            _map = lambda elms: [e.period for e in elms]
            return [_map(s.elements) for s in self._structures]
        elif descriptor == 'mass': 
            _map = lambda elms: [e.mass for e in elms]
            return [_map(s.elements) for s in self._structures]
        elif descriptor == 'electronegativity': 
            _map = lambda elms: [get_element(e).electronegativity for e in elms]
            return [_map(s.elements) for s in self._structures]
        elif descriptor == 'electron_affinity': 
            _map = lambda elms: [get_element(e).electron_affinity for e in elms]
            return [_map(s.elements) for s in self._structures]

        elif descriptor == 'atomic_radius': 
            _map = lambda elms: [get_element(e).atomic_radius for e in elms]
            return [_map(s.elements) for s in self._structures]
        elif descriptor == 'covalent_radius': 
            _map = lambda elms: [get_element(e).covalent_radius for e in elms]
            return [_map(s.elements) for s in self._structures]
        elif descriptor == 'ionic_radius': 
            _map = lambda elms: [get_element(e).ionic_radius for e in elms]
            return [_map(s.elements) for s in self._structures]
        elif descriptor == 'vdw_radius': 
            _map = lambda elms: [get_element(e).vdw_radius for e in elms]
            return [_map(s.elements) for s in self._structures]

class PropertyDescriptorBuilder(DescriptorBuilder):
    
    def __init__(self, structures):
        # super().__init__(structures=structures)
        self.descriptor=PropertyDescriptor(structures=structures)
        self.descriptor._builder=self

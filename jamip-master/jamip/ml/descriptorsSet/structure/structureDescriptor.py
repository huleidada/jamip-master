# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from ..descriptor import Descriptor, DescriptorBuilder
from ase.atoms import Atoms

def jp2ase(structure):

    atom_symbols = [atom.element.symbol for atom in structure.atoms]
    atoms_pos = [atom.position for atom in structure.atoms]
    basis_vectors = structure.lattice    
    atoms = Atoms(symbols=atom_symbols, cell=basis_vectors, pbc=True)
    atoms.set_scaled_positions(atoms_pos)
    return atoms

def zero_pad(self, array):
    """Zero-pads the given matrix.

    Args:
        array (np.ndarray): The array to pad

    Returns:
        np.ndarray: The zero-padded array.
    """
    import numpy as np
    # Pad with zeros
    n_atoms = array.shape[0]
    n_dim = array.ndim
    padded = np.pad(array, [(0, self.n_atoms_max-n_atoms)]*n_dim, 'constant')

    return padded



class StructureDescriptor(Descriptor):
    
    _params_ = {
       'mbtr': {
          'species': ['C'],
          'periodic': True,
          'k1': None,
          'k2': None,
          'k3': None,
          'flatten': False,
          'sparse': False,
          'normalization': "l2_each",
       },
       'soap': {
         "species": ['C'],
         "periodic": True,
         "rcut": 6,
         "nmax": 2,
         "lmax": 0,
         "sigma": 1.0, 
         "rbf": 'gto',
         "crossover": True, 
         "average": 'off',
         "sparse": False,
       }, 
       'acsf':{
          "species": ["C"],
          "rcut": 6.0,
          "g2_params": None,
          "g2_params": None,
          "g4_params": None,
          "g4_params": None,
          'periodic': True,
          'sparse': False,
       },
       'sine': {
          "n_atoms_max" : 1, 
          "permutation": "sorted_l2", 
          "sigma": None, 
          "seed": None, 
          "flatten": False, 
          "sparse": False
       },
       'ewald': {
          "n_atoms_max" : 1, 
          "permutation": "sorted_l2", 
          "sigma": None, 
          "seed": None, 
          "flatten": False, 
          "sparse": False
       },
       'cm': {
          "n_atoms_max" : 1, 
          "permutation": "sorted_l2", 
          "sigma": None, 
          "seed": None, 
          "flatten": False, 
          "sparse": False
       },
    }

    def __init__(self, structures):
        super().__init__(structures=structures)
        self._model = None
        
    def _getValues(self, descriptor):

        return self.run(descriptor)


    def _getModel(self, model, **kwargs):
        from copy import deepcopy
        import warnings
        if model in self._params_:
            params = deepcopy(self._params_[model])
            params.update(kwargs)
        else:
            msg = 'unsupport input model %s' %model
            raise KeyError(msg)

        if model == 'mbtr':
            from dscribe.descriptors import MBTR
            # check if vaild input %
            if not any([params[key] for key in ['k1','k2','k3']]):
                msg='model %s stop for missing key parameters' %model
                raise RuntimeError(msg)
            else:
                Model = MBTR(**params)

        elif model == 'soap':
            from dscribe.descriptors import SOAP
            if params['rcut'] == 6 and params['nmax']==2 and params['lmax']==0:
                warnings.warn('SOAP model uses default parameters, please make sure your settings are correct! ')
            Model = SOAP(**params)

        elif model == 'acsf':
            from dscribe.descriptors import ACSF
            if not any([params[key] for key in ['g2_params','g3_params','g4_params','g5_params']]):
                warnings.warn('ACSF model uses default parameters, please make sure your settings are correct! ')
            Model = ACSF(**params)

        elif model == 'cm':
            from dscribe.descriptors import CoulombMatrix
            Model = CoulombMatrix(**params)

        elif model == 'sine':
            from dscribe.descriptors import SineMatrix
            Model = SineMatrix(**params)

        elif model == 'ewald':
            from dscribe.descriptors import EwaldSumMatrix
            Model = EwaldSumMatrix(**params)

        return Model

    @property
    def model(self):
        return self._model

    @model.deleter
    def model(self):
        del self._model
    
    def set_model(self, model, **kwargs):
        if model not in self._params_:
            msg='unsupport input model %s' %model
            raise KeyError(msg)
        else:
            self._params_[model].update(kwargs)
            self._model = model

        return self

    def run(self,model):

        outputs = []
        Model = self._getModel(model)
        if model in ['mbtr','acsf','soap']:
            for structure in self._structures:
                Model.species = [element.symbol for element in structure.elements]
                outputs.append(Model.create(jp2ase(structure)))
        elif model in ['cm','sine','ewald']:
            Model.n_atoms_max = max([s.natoms for s in self._structures])
            for structure in self._structures:
                outputs.append(Model.create(jp2ase(structure)))
            
        return outputs

    def mbtr(self,structures):
        from dscribe.descriptors import MBTR
        model = MBTR(
            species=[elm.symbol for elm in structure.elements],
            k1={
                "geometry": {"fuction": "atomic_number"},
                "grid": {"min": 0, "max": 8, "n": 100, "sigma": 0.1},
            },
            k2={
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.1},
                "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
            },
            k3={
                "geometry": {"function": "cosine"},
                "grid": {"min": -1, "max": 1, "n": 100, "sigma": 0.1},
                "weighting": {"function": "exponential", "scale": 0.5, "cutoff": 1e-3},
            },
        )

class StructureDescriptorBuilder(DescriptorBuilder):
    
    def __init__(self, structures):
        self.descriptor=StructureDescriptor(structures=structures)
        self.descriptor._builder=self

    def set_model(self, model, **kwargs):
        self.descriptor.set_model(model,**kwargs)
        return self
         
    def run(self):
        if self.descriptor._model == None:
            msg='no active model was found'
            raise KeyError(msg)
        
        self.descriptor._feature = self.descriptor.run(self.descriptor._model)

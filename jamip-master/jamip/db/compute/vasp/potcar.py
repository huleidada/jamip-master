# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
from collections import OrderedDict
from utils.variables import environment_variables

class POTCAR:
    def __init__(self, dtype, path, filename='POTCAR', **kwargs):
        """
        Arguments:
            dtype: type of exchange-correlation functional. i.e. 'LDA', 'PBE', 'PW91'
            path: the path of outputing.
            filename (default='POTAR'): filename of INCAR file.
            
            kwargs:
                pseduopotential_of_elements: {'Na': 'Na', 'Cs': 'Cs_pv',...}. Note that, in Python 3.7, 'Dict keeps insertion order' is the ruling. 
                parameters: 
        """
        self.path=path
        self.filename=filename
        
        path_of_pseduopotential=None
        if dtype.lower() == 'lda':
            path_of_pseduopotential='{}/LDA'.format(environment_variables.pseudopotential_of_VASP.value)
        elif dtype.lower() == 'pbe':
            path_of_pseduopotential='{}/PBE'.format(environment_variables.pseudopotential_of_VASP.value)
        elif dtype.lower() == 'pw91':
            path_of_pseduopotential='{}/PW91'.format(environment_variables.pseudopotential_of_VASP.value)
        else:
            raise ValueError('unknown dtype')
            
        self.path_of_pseduopotential=path_of_pseduopotential
#        self.pseduopotential_of_elements=OrderedDict() if not('pseduopotential_of_elements' in kwargs) else kwargs['pseduopotential_of_elements']
        self.parameters=OrderedDict()
        if 'parameters' in kwargs: self.add_parameters(kwargs['parameters'])
            
    def add_parameters(self, parameters):
        """
        add parameters of POTCAR.
        
        Arguments:
            parameters: dictionary array. i.e. {'Na': 'Na', 'Cs': 'Cs_pv',...}
            
        Return:
            POTCAR's object.
        """
        for key, value in parameters.items():
            
            import warnings
            warnings.warn('check the legality of parameter')
            """
            do something
            """
            
            self.parameters[key]=value
        return self
    
    def remove_parameters(self, parameters):
        """
        remvoe parameters from POTCAR.
        
        Arguments:
            parameters: list array. i.e. ['Na', 'Cs', ...]
            
        Return:
            POTCAR's object.
        """
        for p in parameters:
            if p in self.parameters.keys():
                self.parameters.pop(p)
            else:
                raise ValueError("Doesn't exist in parameters")
        
        return self
    
    def set_parameters(self, parameters):
        """
        set parameters of POTCAR. Note that it will clear all old parameters firstly before set parameters.
        
        Arguments:
            parameters: dictionary array. i.e. {'Na': 'Na', 'Cs': 'Cs_pv',...}
            
        Return:
            POTCAR's object.
        """
        self.parameters=OrderedDict()
        self.add_parameters(parameters)
        return self
        
    def output(self):
        """
        output POTCAR file.
            
        Return:
            POTCAR's object.
        """
        path=self.path
        filename=self.filename
        
# =============================================================================
#         with open('{}/{}'.format(path, filename), 'w') as f:
#             f.write('')
#         return self
# =============================================================================
        # check
        potcar='{}/{}'.format(path, filename)
        if os.path.exists(potcar): os.system('rm {}'.format(potcar))
            
        for key, value in self.parameters.items():
            pseduopotential='{}/{}/POTCAR'.format(self.path_of_pseduopotential, value)
            if os.path.exists(pseduopotential):
                os.system('cat {} >> {}'.format(pseduopotential, potcar))
            else:
                raise ValueError('non-existed pseduopotential in parameters')
        return self
            
            

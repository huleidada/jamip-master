# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from collections import OrderedDict


class INCAR:
    
    def __init__(self, path, filename='INCAR', **kwargs):
        
        """
        if type and parameters simultaneously exist in the list of parameters, 
        firstly load the predefined parameters in 'type', then load that in 'parameters'.
        
        Arguments:
            path: the path of outputing.
            filename (default='INCAR'): filename of INCAR file.
            
            kwargs:
                type: template parameters. i.e. 'scf', 'opt', etc.
                parameters: customized parameters (dictionary-type). i.e. {'EDIFF':1e-6, 'EDIFFG':-1e-4, ...}
        """
        self.path=path
        self.filename=filename
        self.parameters=OrderedDict()
        
        if 'type' in kwargs:
            if kwargs['type'].lower() == 'scf':
                self.set_parameters(scf)
        if 'parameters' in kwargs:
            self.add_parameters(kwargs['parameters'])
        
    def add_parameters(self, parameters):
        """
        add parameters of INCAR.
        
        Arguments:
            parameters: dictionary array. i.e. {'EDIFF':1e-6, 'EDIFFG':-1e-4, ...}
            
        Return:
            INCAR's object.
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
        remvoe parameters from INCAR.
        
        Arguments:
            parameters: list array. i.e. ['EDIFF', 'EDIFFG', ...]
            
        Return:
            INCAR's object.
        """
        for p in parameters:
            if p in self.parameters.keys():
                self.parameters.pop(p)
            else:
                raise ValueError("Doesn't exist in parameters")
        
        return self
    
    def set_parameters(self, parameters):
        """
        set parameters of INCAR. Note that it will clear all old parameters firstly before set parameters.
        
        Arguments:
            parameters: dictionary array. i.e. {'EDIFF':1e-6, 'EDIFFG':-1e-4, ...}
            
        Return:
            INCAR's object.
        """
        self.parameters=OrderedDict()
        self.add_parameters(parameters)
        return self
    
    def output(self):
        """
        output INCAR file.
            
        Return:
            INCAR's object.
        """
        path=self.path
        filename=self.filename
        parameters=self.parameters
        with open('{}/{}'.format(path, filename), 'w') as f:
            for key, value in parameters.items():    
                f.write('{} = {}\n'.format(key, value))
        return self


# predefined parameters for different type of calculations
relax={'ISTART': 0,
       'ICHARG': 2,
       'PREC': 'Accruate',
       'LREAL': '.FALSE.',
       'NSW': 500,
       'IBRION': 2,
       'POTIM': 0.2,
       'ISIF': 3,
       'EDIFF': 1e-6,
       'EDIFFG': -0.005,
       'ENCUT': 500,
       'ISMEAR': 0,
       'SIGMA': 0.05,
       'LWAVE': '.FALSE.',
       'LCHARG': '.FALSE.',
       }    
    
scf={'ISTART': 1,
     'ICHARG': 2,
     'PREC': 'Accruate',
     'LREAL': '.FALSE.',
     'NSW': 500,
     'IBRION': 0,
     'EDIFF': 1e-6,
     'ENCUT': 500,
     'ISMEAR': 0,
     'SIGMA': 0.05,
     'LWAVE': '.FALSE.',
     'LCHARG': '.FALSE.'
     }
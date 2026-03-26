# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from collections import OrderedDict


class OPTCELL:
    def __init__(self, path, filename='OPTCELL', **kwargs):
        """
        Arguments:
            path: the path of outputing.
            filename (default='KPOINTS'): filename of INCAR file.
        """
        self.path=path
        self.filename=filename
        
        self.optcell=None
        
        if 'optcell' in kwargs:
            self.set_optcell(optcell=kwargs['optcell'])
        
    def set_optcell(self, optcell):
        """
        set optcell
        
        Arguments:
            optcell: fixed information. i.e. '110' -> fix z; '011' -> fix x
            
        Return:
            OPTCELL's object.
        """
        self.optcell=optcell
        
    def output(self):
        """
        output POTCAR file.
            
        Return:
            INCAR's object.
        """
        path=self.path
        filename=self.filename
        
        with open('{}/{}'.format(path, filename), 'w') as f:
            if not self.optcell is None:
                f.write('{}'.format(self.optcell))
            else:
                raise ValueError('optcell is None')
        return self

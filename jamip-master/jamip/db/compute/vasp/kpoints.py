# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from collections import OrderedDict


class KPOINTS:
    def __init__(self, path, filename='KPOINTS', **kwargs):
        """
        Arguments:
            path: the path of outputing.
            filename (default='KPOINTS'): filename of INCAR file.
        """
        self.path=path
        self.filename=filename
        
        self.paths=OrderedDict()
        
        
    def output(self):
        """
        output POTCAR file.
            
        Return:
            KPOINTS's object.
        """
        path=self.path
        filename=self.filename
        
        with open('{}/{}'.format(path, filename), 'w') as f:
            f.write('')
        return self

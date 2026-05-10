# -*- coding: utf-8 -*-
#!/usr/bin/env python3


class POSCAR:
    def __init__(self, path, filename='POSCAR', **kwargs):
        """
        Arguments:
            path: the path of outputing.
            filename (default='POSAR'): filename of INCAR file.
        """
        self.path=path
        self.filename=filename
        
        self.structure=None
        if 'structure' in kwargs:
            self.set_structure(structure=kwargs['structure'])
        
    def set_structure(self, structure):
        """
        Arguments:
            structure: structure's object.
        """
        from materials.structure import Structure
        
        # ckeck
        if not isinstance(structure, Structure):
            raise ValueError('unknown structure')
            
        self.structure=structure
        return self
    
    def output(self):
        """
        output POSCAR file.
            
        Return:
            POSCAR's object.
        """
        from iostream.write import Write
        
        path=self.path
        filename=self.filename
        
        if not(self.structure is None):
            Write(structure=self.structure, path='{}/{}'.format(path, filename), dtype='poscar').run()
        else:
            raise ValueError('non-existed structure')
        
# =============================================================================
#         with open('{}/{}'.format(path, filename), 'w') as f:
#             f.write('')
# =============================================================================
        return self

# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
from jamip.db.utils.variables import default_constants
from ..structureFactory import StructureFactory

class AdvancedStructureFactory(StructureFactory):
    
    def __init__(self, structure, isOperateOnSelf=False, isPersist=False, **kwargs):
        """
        Arguments:
            structure: structure's object.
            isOperateOnSelf: Whether to operate itself.
            isPersist (default=False): whether to save to the database.
            
            kwargs:
                isCloneFullInfo (default=False): whether to clone all information of structure.
        """
        super(AdvancedStructureFactory, self).__init__(structure=structure, 
                                         isOperateOnSelf=isOperateOnSelf, 
                                         isPersist=isPersist,
                                         **kwargs)
        
    def moire_pattern(self):
        pass
    
    def twin():
        pass
    
    def bending():
        pass

# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from compute.check import Check
from compute.vasp.calculateException import CalculateException

class Check4VASP(Check):
    """
    check the validity of the result in the VASP calculation.
    """
    
    def run(self):
#        raise ValueError
        raise CalculateException

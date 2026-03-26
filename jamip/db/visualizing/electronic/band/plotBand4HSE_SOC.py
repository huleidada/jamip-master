#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math
import collections
from progressbar import ProgressBar, Percentage, Bar, ETA
from plotBand import PlotBand

class PlotBand4HSE_SOC(PlotBand):
    def __init__(self, isSemi, isDecomposed, path=None):
        super().__init__(isSemi=isSemi, isDecomposed=isDecomposed, path=path)
    
    def getNeededKpoint(self, string, kpoints, precision=1e-6):
        kp=self.getKpoint(string)
        weight=float(string.split()[-1])
        if np.abs(weight-0) <= precision:
            if kpoints == []:
                kpoints=kp
            else:
                kpoints=np.vstack([kpoints, kp])
        return kpoints
    
    def getBandEnergy(self, string, Efermi, bk, precision=1e-6):
        """
        read band energy from a string.
        
        Arguments:
            string:
            Efermi:
            
        Return:
            
        """
        temp=np.array(float(string.split()[4]) - Efermi)
        weight=float(string.split()[-1])
        if np.abs(weight-0) <= precision:
            if bk == []:
                bk=temp
            else:
                bk=np.vstack([bk, temp])
        return bk
    
# ------------------- test ------------------
p=PlotBand4HSE_SOC(isSemi=True, isDecomposed=True, path='/home/fu/workspace/Thallide/TlBi5Te8/opt/2/3/mbj+soc')
print(p.getLatticeVector())
print(p.getNeededKpoint(string=' k-point     1 :    0.00000000 0.00000000 0.00000000     weight = 0.0000000', 
                        kpoints=[]))     
    
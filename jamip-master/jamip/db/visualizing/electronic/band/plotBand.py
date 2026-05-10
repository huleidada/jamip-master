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
from abc import ABCMeta, abstractmethod


class PlotBand:
    __metaclass__=ABCMeta
    
    def __init__(self, isSemi, isDecomposed, path=None):
        """
        Arguments:
            isSemi: whether it is a semiconductor (True/False). If true, need to shit the Fermi level to the VBM when plotting band.
            isDecomposed: whether to decompose atomic orbits.            
        """
        if path is None:
            self.path=os.getcwd();
        else:
            self.path=path
            
        self.isSemi=isSemi
        self.isDecomposed=isDecomposed
        
    def getLatticeVector(self, filename='POSCAR'):
        """
        get the lattice vector and reciprocal lattice vector.
        """
        infile=open('{}/{}'.format(self.path, filename))
        string=infile.readline() # comment line

        string=infile.readline()
        latticeConstant=float(string)
        
        a=[] # lattice vector
        for i in range(0, 3):
            string=infile.readline()
            temp=np.array([float(s0)*latticeConstant for s0 in string.split()])
            if (a == []):
                a=temp
            else:
                a=np.vstack([a, temp])

        volume=a[0][0]*a[1][1]*a[2][2]+a[0][1]*a[1][2]*a[2][0]+ \
               a[0][2]*a[1][0]*a[2][1]-a[0][0]*a[1][2]*a[2][1]- \
               a[0][1]*a[1][0]*a[2][2]-a[0][2]*a[1][1]*a[2][0]
        
        b=np.zeros((3,3)) # reciprocal lattice vector
        for i in range(0,3):
            if i == 0:
                j=1
                k=2
            elif i == 1:
                j=2
                k=0
            else:
                j=0
                k=1
            c=np.zeros(3)
            c[0]=a[j][1]*a[k][2]-a[j][2]*a[k][1]
            c[1]=a[j][2]*a[k][0]-a[j][0]*a[k][2]
            c[2]=a[j][0]*a[k][1]-a[j][1]*a[k][0]
            for j in range(0, 3):
                b[i][j]=2*math.pi*c[j]/volume
       
        return a, b, volume
    
    def getEfermi(self, filename='OUTCAR'):
        """
        get Fermi energy level from OUTCAR.
        """
        infile=open('{}/{}'.format(self.path, filename))
        string=os.popen("grep 'E-fermi' %s" %infile).readline()
        Efermi=float(string.split()[2])
        return Efermi
    
    def getKpointsNum(self, filename='KPOINTS'):
        """
        get number of kpoints from KPOINTS file.
        
        Note that: remove the blank line at end of the file if exist.
        """
        infile=open('{}/{}'.format(self.path, filename))
        num_per_path=int(os.popen('head -2 %s | tail -1' %infile).readline())
        num_of_paths=(int(os.popen('wc -l %s' %infile).readline().split()[0])-4)/2
        return num_per_path*num_of_paths
    
    def getKpoint(self, string):
        """
        read kpoint from a string.
        
        Arguments:
            string: 
        
        Return:
            kpoint's array
        """
        #kp=np.array([float(s0) for s0 in string.split()[3:6]])
        tmp=string.split(':')[1].split('weight')[0].split()
        if len(tmp) == 3:
            kp=np.array([float(s0) for s0 in tmp])
        else: # kpoint without blank
            newtmp=[]
            for component in tmp:
                isign=[pos for pos, char in enumerate(component) if char == '-'] # index of sign (-)
                if len(isign) == 0:
                    newtmp.append(component)
                elif len(isign) == 1:
                    if isign[0] == 0:
                        newtmp.append(component)
                    else:
                        newtmp.append(component[:isign[0]])
                        newtmp.append(component[isign[0]:])
                elif len(isign) == 2:
                    if isign[0] == 0:
                        newtmp.append(component[:isign[1]])
                        newtmp.append(component[isign[1]:])
                    else:
                        newtmp.append(component[:isign[0]])
                        newtmp.append(component[isign[0]:isign[1]])
                        newtmp.append(component[isign[1]:])
                elif len(isign) == 3:
                    if isign[0] != 0:
                        raise ValueError('unknown kpoint')
                    newtmp.append(component[:isign[1]])
                    newtmp.append(component[isign[1]:isign[2]])
                    newtmp.append(component[isign[2]:])
            kp=np.array([float(s0) for s0 in newtmp])
        return kp
    
    def getNeededKpoint(self, string, kpoints):
        kp=self.getKpoint(string)
        if kpoints == []:
            kpoints=kp
        else:
            kpoints=np.vstack([kpoints, kp])
        return kpoints
    
    def getBandEnergy(self, string, Efermi, bk):
        """
        read band energy from a string.
        
        Arguments:
            string:
            Efermi:
            
        Return:
            
        """
        temp=np.array(float(string.split()[4]) - Efermi)
        if bk == []:
            bk=temp
        else:
            bk=np.vstack([bk, temp])
        return bk
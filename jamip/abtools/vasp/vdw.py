# coding: utf-8
# Copyright (c) JAMIP Development Team.
# Distributed under the terms of the JLU License.

#=================================================================
# This file is part of JAMIP.
#
# Copyright (C) 2021 Jilin University
#
#  JAMIP is a platform for high throughput calculation. It aims to 
#  make simple to organize and run large numbers of tasks on the 
#  superclusters and post-process the calculated results.
#  
#  JAMIP is a useful packages integrated the interfaces for ab initio 
#  programs, such as, VASP, Guassian, QE, Abinit and 
#  comprehensive workflows for automatically calculating by using 
#  simple parameters. Lots of methods to organize the structures 
#  for high throughput calculation are provided, such as alloy,
#  heterostructures, etc.The large number of data are appended in
#  the MySQL databases for further analysis by using machine 
#  learning.
#
#  JAMIP is free software. You can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published 
#  by the Free sofware Foundation, either version 3 of the License,
#  or (at your option) and later version.
# 
#  You should have recieved a copy of the GNU General Pulbic Lincense
#  along with JAMIP. If not, see <https://www.gnu.org/licenses/>.
#=================================================================

#from periodictable import elements

__contributor__ = 'Guangren Na, Xingang Zhao'
__update_date__ = '2017.05.01'

"""SetVdw:: partial of JAMIP for vdw functional setting, default is None."""
from typing import Union
import logging

class VDW(object):
    """
    Common Van der Waals functional implemented in VASP:
        --D2: 
        --optB86b:
        --optB88:
        --DF2:
        --optPBE:
        --revPBE:
        --revDF2:
        --rVV10:
    args::
        vdw:: a string to describe the vdw functional name;
        elements:: species only for D2 functional;

    :return: A parameter dict.
    """
     
    def __init__(self):
	
        self.__vdw = None
        self.vdWdat = None
        self.vdWdeny = []

    @property
    def van_der_Waals(self):
        vdw_dict = {'d2': 'D2', 
                    'd3': 'D3', 
                    'b86': 'B86', 
                    'b88': 'B88',
                    'pbe': 'PBE',
                    'df2': 'DF2',
                    'revdf2': 'rDF2',
                    'rvv10': 'rVV10',
                    'revpbe': 'rPBE',
                    'optpbe': 'oPBE'
                   }
        return vdw_dict

    @property
    def vdw(self):

        return self.__vdw

    @vdw.setter
    def vdw(self, value:Union[str,tuple]):
        """
        Optional parameters
        > vasp.vdw = "b86"
        > vasp.vdw = "b86", "/public/apps/vasp/vdw.dat"
        """
        from jamip.utils.logger import full_path
        if isinstance(value, str):
            vdw = value.lower()
        elif len(value) == 2:
            vdw = value[0].lower()
            self.vdWdat = full_path(value[1])
        else:
            logging.error("Invalid input: vasp.vdw")
        
        # vdW type
        if vdw in self.van_der_Waals:
            self.__vdw = self.van_der_Waals[vdw]
        else:
            logging.error('This vdW function is not yet supported')

        if self.__vdw in ['B86', 'B88', 'DF2', 'rDF2', 'rPBE','oPBE', 'rVV10']:
            if self.vdWdat == None:
                logging.error('vdw_kernel.bindat is necessary! Please add path after vasp.vdw. ')

    @vdw.deleter
    def vdw(self):
         
        del self.__vdw

    def __d2vdw__(self, species):

        from jamip.structure.atomic_number import number

        c6 = [ 0.14 ,  0.08 ,  1.61 ,  1.61 ,  3.13 ,  1.75 ,  1.23 ,  0.7  ,
         0.75 ,  0.63 ,  5.71 ,  5.71 , 10.79 ,  9.23 ,  7.84 ,  5.57 ,
         5.07 ,  4.61 , 10.8  , 10.8  , 10.8  , 10.8  , 10.8  , 10.8  ,
        10.8  , 10.8  , 10.8  , 10.8  , 10.8  , 10.8  , 16.99 , 17.1  ,
        16.37 , 12.64 , 12.47 , 12.01 , 24.67 , 24.67 , 24.67 , 24.67 ,
        24.67 , 24.67 , 24.67 , 24.67 , 24.67 , 24.67 , 24.67 , 24.67 ,
        37.32 , 38.71 , 38.44 , 31.74 , 31.5  , 29.99 ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ]

        r0 = [ 1.001,  1.012,  0.825,  1.408,  1.485,  1.452,  1.397,  1.342,
         1.287,  1.243,  1.144,  1.364,  1.716,  1.716,  1.705,  1.683,
         1.639,  1.595,  1.485,  1.474,  1.562,  1.562,  1.562,  1.562,
         1.562,  1.562,  1.562,  1.562,  1.562,  1.562,  1.65 ,  1.727,
         1.76 ,  1.771,  1.749,  1.727,  1.628,  1.606,  1.639,  1.639,
         1.639,  1.639,  1.639,  1.639,  1.639,  1.639,  1.639,  1.639,
         1.672,  1.804,  1.881,  1.892,  1.892,  1.881,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
         0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ]

        R0 = []
        C6 = []
	
        # need to updated % D2 not enough %

        for elm in species:
            id = number[elm]
            if id > 55: print('Warning: no R0 and C6 values for '+ elm)
		    
            R0.append(r0[id])
            C6.append(c6[id])

        return  {\
            "ivdw": 1, \
            "vdw_r0": ' '.join(str(r) for r in R0), \
            "vdw_c6": ' '.join(str(c) for c in C6)}
            

    def get_vdW_params(self, elements=None):

        value = self.vdw

        # case d2 vdw %
        if value == 'D2':
            return self.__d2vdw__(elements)

        if value == 'D3':
            return {'ivdw': 11}

        # case PBE vdw % 
        if value == 'PBE':
            return {\
                'luse_vdw': True, \
                "lasph":True, \
                'aggac': 0.0000}

        # case optPBE vdw % 
        if value == 'oPBE':
            return {\
                'gga': 'OR', \
                'luse_vdw': True, \
                "lasph":True, \
                'aggac': 0.0000}
                

        # for optvdw-B88 %
        if value == 'B88':
            return {\
                'gga': 'BO', \
                'luse_vdw': True, \
                "lasph":True, \
                'aggac': 0.0000, \
                'param1': 0.1833333333, \
                'param2': 0.2200000000}
                

        # for optvdw-B86b %
        if value == 'B86':
            return {\
                'gga': 'MK', \
                'luse_vdw': True, \
                "lasph":True, \
                'param1': 0.1234, \
                'param2': 1.0000, \
                'aggac': 0.0000}
                

        # for DF2 %
        if value == 'DF2':
            return {\
                "gga": 'ML', \
                "luse_vdw": True, \
                "aggac": 0.0000, \
                "lasph":True, \
                "zab_vdw": -1.8867}
                
        # for revDF2 %
        if value == 'rDF2':
            return {\
                "gga": 'MK', \
                "luse_vdw": True, \
                "aggac": 0.0000, \
                "lasph":True, \
                'param1': 0.1234, \
                'param2': 0.711357, \
                "zab_vdw": -1.8867}
                
        # for revPBE + vdW-DF %
        if value == 'rPBE':
            return {\
                "gga": 'RE', \
                "luse_vdw": True, \
                "aggac": 0.0000, \
                "lasph": True}
                
        # for scan+rvv10 %
        if value == 'rVV10':
            return {\
                "gga": 'SCAN', \
                "luse_vdw": True, \
                "bparam": 15.7, \
                "lasph": True}
                

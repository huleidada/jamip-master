# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from enum import Enum
from functools import lru_cache
import pandas as pd
import os
#from mendeleev import element as me

module_dir = os.path.dirname(os.path.abspath(__file__))
IonicRadius = pd.read_csv(os.path.join(module_dir, "ionic.csv"))

def int2Roman(num):

    c={0:("","I","II","III","IV","V","VI","VII","VIII","IX"),
       1:("","X","XX","XXX","XL","L","LX","LXX","LXXX","XC"),
       2:("","C","CC","CCC","CD","D","DC","DCC","DCCC","CM"),
       3:("","M","MM","MMM")}
    roman=[]
    roman.append(c[3][num//1000%10])  
    roman.append(c[2][num//100%10])  
    roman.append(c[1][num//10%10])  
    roman.append(c[0][num%10])
    return ''.join(roman)

@lru_cache()
def getIonicRadius(elm,charge,coordination,ls=None):

    screen = []
    ir0 = []
    ir1 = []
    if isinstance(coordination,int): 
        coordination = int2Roman(coordination)
    for ir in IonicRadius.values:
        if ir[0] == elm:
            ir0.append(ir)
            if ir[1] == charge:
                ir1.append(ir[2])
                if ir[2] == coordination:
                    screen.append(ir)

    if len(screen) == 1:
        return screen[0][3]
    elif len(screen) == 0 and len(ir1) > 0:
        msg = 'Please Try these optional :'+' '.join(ir1)
        raise KeyError(msg)
    else:
        print(ir0)
        raise

class property_descriptor(Enum):
    volume_of_structure='volume'
    # mass_of_structure='mass'
    natoms_of_structure='natoms'
    lattice_of_structure='lattice'
    spacegroup_of_structure='spacegroup'
    composition_of_structure='composition'

    a_of_lattice='a'
    b_of_lattice='b'
    c_of_lattice='c'
    alpha_of_lattice='alpah'
    beta_of_lattice='beta'
    gamma_of_lattice='gamma'

    min_bond_length='BX'

    symbol_of_elements='symbol'
    z_of_elements='z'
    group_of_elements='group'
    period_of_elements='period'
    mass_of_elements='mass' 
    electronegativity_of_elements = 'electronegativity'    
    electron_affinity_of_elements = 'electron_affinity'

    covalent_radius_of_elements = 'covalent_radius'
    atomic_radius_of_elements = 'atomic_radius'
    ionic_radius_of_elements = 'ionic_radius'
    vdw_radius_of_elements = 'vdw_radius'
    
    
class structure_descriptor(Enum):
    bcm='bcm'
    soap='soap'
    mbtr='mbtr'
    Coulomb_Matrix='cm'
    Sine_matrix='sine'
    Ewald_sum_matrix='ewald'
    Atom_centered_Symmetry_Functions="ascf"
    Smooth_Overlap_of_Atomic_Positions="soap"
    Many_body_Tensor_Representation="mbtr"

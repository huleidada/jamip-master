# -*- coding: utf-8 -*-

from .variables import property_descriptor, structure_descriptor

def isPropertyDescriptor(descriptor):
    result=False
    for pd in property_descriptor:
        if descriptor == pd.value:
            result=True
            break
        
    return result

def isStructureDescriptor(descriptor):
    result=False
    for sd in structure_descriptor:
        if descriptor == sd.value:
            result=True
            break
        
    return result
    
def isStructureWyckoff(structure,wyckoff=None):
    import numpy as np
    wycdict = {}
    for value in structure.wyckoffSites.values():
        symbol = value['symbol']
        multi = value['multiplicity']
        wycdict[symbol] = multi
    wycsymbol = ''
    div = np.gcd.reduce(list(wycdict.values()))
    for key in sorted(wycdict):
        wycsymbol += key
        value = wycdict[key]/div
        if value > 1:
           wycsymbol += "%d" %value

    if wycsymbol == wyckoff:
        return True
    else:
        return wycsymbol

            

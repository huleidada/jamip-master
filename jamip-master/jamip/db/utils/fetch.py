'''
Created on Dec 11, 2017

@author: Yuhao Fu
'''
import numpy as np

    
def get_index_from_positions(collection_of_positions, position, **kwargs):
    """
    get the index of given poistion from the collection (used for addressing the dataset of spglib).
        
    Arguments:
        collection_of_: collection of positions. Note that the type is 'Direct'.
        position: entity's index or object.
        
        kwargs:
            precision (default=1e-3): used to determine whether the two atoms are overlapped. Note that, 
                to determine whether this atom is in collection by comparing its distance 
                from other atoms.
            for "Cartesian"-type position:
                lattice: lattice parameter of structure. i.e.
                    [[x1,y1,z1],
                     [x2,y2,z2],
                     [x3,y3,z3]]
                    
    Return:
        index if exist. Conversely, False.
    """
    from .check import check_formated_position_only_direct, check_formated_position_only_cartesian
    from .convert import cartesian2direct
    
    index=None    

    precision=1e-3
    if 'precision' in kwargs:
        precision=kwargs['precision']
                    
    if check_formated_position_only_cartesian(position):
        if 'lattice' not in kwargs:
            raise ValueError('lattice must be given')
        lattice=kwargs['lattice']
        position=cartesian2direct(lattice, position)
    elif check_formated_position_only_direct(position):
        # collection=np.array(collection_of_positions) if not isinstance(collection_of_positions, np.ndarray) else collection_of_positions
        collection=np.asarray(collection_of_positions)
        for i in range(0, collection.shape[0]):
            distance=collection[i]-np.array(position)
                
            # note the periodic boundary condition. like [0.0, 0.0, 0.0] vs. [0.999999, 0.0, 0.0]
            for j in range(0, len(distance)):
                if distance[j] > 0.5:
                    distance[j]=1-distance[j]
                elif distance[j] < -0.5:
                    distance[j]=1+distance[j]
                
            if np.linalg.norm(distance) <= precision:
                if index is None:
                    index=i
                else:
                    raise ValueError('exist reduplicative positon')

    return index

def get_time():
    """
    Returns:
        current time.
    """
    import time
    
    return time.strftime("%m-%d-%Y %H:%M:%S:", time.localtime(time.time()))


def get_atoms_from_cell(cell):
    """
    fetch atoms from cell-type structure.
    
    Arguments:
        cell: cell-type structure. The valid format is:
            {'lattice':lattice,
             'positions':positions,
             'numbers':numbers,
             'magmoms':magmoms(optional)}
        
    Returns:
        collection of atoms.
    """
    positions=cell['positions']
    numbers=cell['numbers'] # z of atoms
    
    atoms=[]
    for i in range(0, len(numbers)):
#        symbol=Element.objects.filter(z=numbers[i])[0].symbol
        symbol=get_symbol_by_z(z=numbers[i])
        atoms.append([symbol]+positions[i].tolist())
    return atoms

def get_symbol_by_z(z):
    """
    get element's symbol by given atomic number (z).
    
    Arguments:
        z: atomic number.
        
    Returns:
        element's symbol if exist. Conversely, None.
    """
    from jamip.structure.atomic_number import atomic
    
    if z in atomic:
        return atomic[z]
    return None

def get_z_by_symbol(symbol):
    """
    get atomic number (z) by element's symbol.
    
    Arguments:
        symbol: element's symbol. i.e. Na
        
    Returns:
        atomic number if exist. Conversely, None.
    """
    from jamip.structure.atomic_number import number
    
    if symbol in number:
        return number[symbol]
    return None

def get_name_by_symbol(symbol):
    """
    get name by element's symbol.
    
    Arguments:
        symbol: element's symbol. i.e. Na
        
    Returns:
        name if exist. Conversely, None.
    """
    from jamip.structure.elementInfo import ElementDict
    
    name = None
    if symbol in ElementDict:
        name = ElementDict[symbol]['name'] 
    return name
        
def get_period_by_symbol(symbol):
    """
    get period by element's symbol.
    
    Arguments:
        symbol: element's symbol. i.e. Na
        
    Returns:
        period if exist. Conversely, None.
    """
    from jamip.structure.elementInfo import ElementDict
    
    if symbol in ElementDict:
        period = ElementDict[symbol]['row'] 
        return int(period)
    return None
        
def get_group_by_symbol(symbol):
    """
    get group by element's symbol.
    
    Arguments:
        symbol: element's symbol. i.e. Na
        
    Returns:
        group if exist. Conversely, None.
    """
    from jamip.structure.elementInfo import ElementDict
    
    if symbol in ElementDict:
        group = ElementDict[symbol]['col'] 
        return int(group)
    return None
            
def get_mass_by_symbol(symbol):
    """
    get atomic mass by element's symbol.
    
    Arguments:
        symbol: element's symbol. i.e. Na
        
    Returns:
        atomic mass if exist. Conversely, None.
    """
    from jamip.structure.elementInfo import ElementDict

    if symbol in ElementDict:
        mass = ElementDict[symbol]['mass'] 
        if not np.isnan(mass):
            return float(mass)
    return None

def get_symbol_from_species(name_of_species):
    """
    get element's symbol from species.
    
    Arguments:
        name_of_species: name of species. i.e. 'Na+', 'Fe3+'
        
    Return:
        element's symbol.
    """
    import re
    from .check import check_species
    
    symbol=None
    if check_species(name_of_species): 
        symbol=name_of_species[:re.search('[0-9.+-]', name_of_species).start()]

    return symbol

def get_oxidation_state_from_species(name_of_species):
    """
    get element's oxidation state from species.
    
    Arguments:
        name_of_species: name of species. i.e. 'Na+', 'Fe3+'
        
    Return:
        element's oxidation state.
    """
    import re
    from .check import check_species
    
    ox=None
    if check_species(name_of_species): 
        symbol=name_of_species[:re.search('[0-9.+-]', name_of_species).start()]
        ox0=name_of_species.replace(symbol, '', 1)
        if ox0 == '+': # i.e. 'Na+'
            ox=1
        elif ox0 == '-': # i.e. 'Cl-'
            ox=-1
        elif ox0.endswith('+'): # i.e. 'Fe3+'
            ox=float(ox0[:-1])
        elif ox0.endswith('-'): # i.e. 'O2-'
            ox=-float(ox0[:-1])

    return ox

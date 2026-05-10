#coding=utf-8
'''
Created on Dec 11, 2017

@author: Yuhao Fu
'''
import numpy as np
from .variables import default_constants

def is_int_or_float(variable):
    """
    check the type of given variable.
    """
    if isinstance(variable, int) or isinstance(variable, float):
        return True
    elif variable.isdigit() or (variable.startswith('+') and (variable.count('+') == 1) and variable.replace('+', '', 1).isdigit()): # i.e. '1' or '+1'
        return True
    elif variable.startswith('-') and (variable.count('-') == 1) and variable.replace('-', '', 1).isdigit(): # '-1'
        return True
    elif ('.' in variable) and (variable.count('.') == 1) and variable.replace('.', '', 1).isdigit(): # i.e. '0.1'
        return True
    elif (variable.startswith('+') and (variable.count('+') == 1)) and (('.' in variable) and (variable.count('.') == 1)) and variable.replace('+', '', 1).replace('.', '', 1).isdigit(): # i.e. '+0.1'
        return True
    elif ((variable.startswith('-') and (variable.count('-') == 1)) and (('.' in variable) and (variable.count('.') == 1))) and variable.replace('-', '', 1).replace('.', '', 1).isdigit(): # '-0.1'
        return True
    else:
        return False

def is_overlap_of_positions(formated_atom1, formated_atom2, lattice, dtype='Direct', **kwargs):
    """
    check whether two given position is overlap.
    
    Arguments:
        formated_atom1:
        formated_atom2: formated atom. Note that type of coordinate is 'Direct',  you can not specify
            the type. The valid formation:
                ['Na', 0.1, 0.0, 0.0, 'Direct']
                ['Na', 0.1, 0.0, 0.0] # Direct-type
                ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                
                contain species information:
                ['Na1+', 0.1, 0.0, 0.0, 'Direct']
                ['Na1+', 0.1, 0.0, 0.0]
                ['Na1+', 5.234, 0.0, 0.0, 'Cartesian']
                
                Also, it is support the following format (Ignore element's symbol).
                [0.1, 0.0, 0.0, 'Direct']
                [0.1, 0.0, 0.0] # Direct-type
                [5.234, 0.0, 0.0, 'Cartesian']
                
        lattice: lattice parameter of structure. i.e.
                [[x1,y1,z1],
                 [x2,y2,z2],
                 [x3,y3,z3]]   
                
        dtype (default='Direct'): type of formated positon after convertion. 'Direct' or 'Cartesian'
          
        kwargs:                
            isNormalizingCoordinate (default=True): whether to remove the periodic boundary condition, 
                    ensure the value of atomic coordinate is between 0 and 1 (i.e. 1.3 -> 0.3).
            precision (default=1e-3): used to determine whether the component of position is very close to 1.
                    from other atoms.
    """
    from .convert import formatting_position
    
    isNormalizingCoordinate=default_constants.isNormalizingCoordinate.value
    if 'isNormalizingCoordinate' in kwargs:
        isNormalizingCoordinate=kwargs['isNormalizingCoordinate']
    precision=default_constants.precision.value
    if 'precision' in kwargs:
        precision=kwargs['precision']
# =============================================================================
#     lattice=None
#     if 'lattice' in kwargs:
#         lattice=kwargs['lattice']
# =============================================================================
    
    position1=formatting_position(formated_atom=formated_atom1, 
                                  dtype=dtype, 
                                  lattice=lattice, 
                                  isNormalizingCoordinate=isNormalizingCoordinate,
                                  precision=precision)
    position2=formatting_position(formated_atom=formated_atom2, 
                                  dtype=dtype, 
                                  lattice=lattice, 
                                  isNormalizingCoordinate=isNormalizingCoordinate,
                                  precision=precision)

    result=None        
    distance=np.array(position1[:3])-np.array(position2[:3])                
    result=True if np.linalg.norm(distance) <= precision else False
    return result

def is_overlap_of_positions_for_molecule(formated_atom1, formated_atom2, **kwargs):
    """
    check whether two given position is overlap.
    
    Arguments:
        formated_atom1:
        formated_atom2: formated atom. The valid formation:
                ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                
                contain species information:
                ['Na1+', 5.234, 0.0, 0.0, 'Cartesian']
                
                Also, it is support the following format (Ignore element's symbol).
                [5.234, 0.0, 0.0, 'Cartesian']
          
        kwargs:                
            precision (default=1e-3): used to determine whether the component of position is very close to 1.
                    from other atoms.
    """    
    precision=default_constants.precision.value
    if 'precision' in kwargs:
        precision=kwargs['precision']
    
    position1=None
    if check_formated_atom_only_cartesian(formated_atom1):
        position1=formated_atom1[1:]
    elif check_formated_position_only_cartesian(formated_atom1):
        position1=formated_atom1
    else:
        raise ValueError('unknown formated_atom1')
    position2=None
    if check_formated_atom_only_cartesian(formated_atom2):
        position2=formated_atom2[1:]
    elif check_formated_position_only_cartesian(formated_atom2):
        position2=formated_atom2
    else:
        raise ValueError('unknown formated_atom2')

    result=None        
    distance=np.array(position1[:3])-np.array(position2[:3])                
    result=True if np.linalg.norm(distance) <= precision else False
    return result

def isOverlap(structure, **kwargs):
    """
    check whether the structure contains overlapping atoms.
    
    Arguments:
        structure : structural object.
        
        kwargs:                
            isNormalizingCoordinate (default=True): whether to remove the periodic boundary condition, 
                    ensure the value of atomic coordinate is between 0 and 1 (i.e. 1.3 -> 0.3).
            precision (default=1e-3): used to determine whether the component of position is very close to 1.
                    from other atoms.
    """
    from copy import deepcopy
    from ..materials.structure import Structure
    from ..materials.molStructure import MolStructure
    
    
    isNormalizingCoordinate=kwargs.get('isNormalizingCoordinate', True)
    precision=kwargs.get('precision', 1e-3)

    formated_atoms=[atom.to_formated_atom() for atom in structure.atoms]
    
    if isinstance(structure, Structure):
        for i,atom1 in enumerate(formated_atoms):
            for atom2 in formated_atoms[i+1:]:
                if is_overlap_of_positions(formated_atom1=atom1,
                                           formated_atom2=atom2,
                                           lattice=structure.lattice, 
                                           isNormalizingCoordinate=isNormalizingCoordinate, 
                                           precision=precision):
                    return True
    elif isinstance(structure, MolStructure):
        for i, atom1 in enumerate(formated_atoms):
            for atom2 in formated_atoms[i+1]:
                if is_overlap_of_positions_for_molecule(formated_atom1=atom1,
                                                        formated_atom2=atom2,
                                                        precision=precision):
                    return True
    else:
        raise ValueError('unknown structure')
    return False
 
def is_inside(position, selective_range, **kwargs):
    """
    chenck whether or not the given position is inside the selectirve range.
    
    Arguments:
        position:
        selective_range: selective range. The valid format is [start, end, direction(0/1/2)]. i.e. [0.24980, 0.31235, 2] or [[0.1, 0.2],
                                                                                                                             [0.1, 0.2],
                                                                                                                             [0.1, 0.2]]
            
        kwargs:                
            tolerance (default=1e-3): tolerance for unit to exclude the right atoms near boundary (unit: Angstrom).
            
            lattice: lattice parameter of structure. i.e.
            [[x1,y1,z1],
             [x2,y2,z2],
             [x3,y3,z3]]
    """
    # check
    if check_formated_position_only_direct(position): 
        pass
    elif check_formated_position_only_cartesian(position):
        from .convert import any2direct
        
        position=any2direct(lattice=kwargs['lattice'], position=position)
    else:
        raise ValueError('unknown position')
    
    # tolerance
    tolerance=kwargs['tolerance'] if 'tolerance' in kwargs else 1e-3
    tolerances = np.ones(3) * tolerance
    
    srange=[]
    if np.array(selective_range).shape == (3,): # i.e. [0.24980, 0.31235, 2] 
        srange=[selective_range[:2] if selective_range[2] == i else [0, 1] for i in range(0, 3)]
    elif np.array(selective_range).shape == (3,2):
        srange=selective_range
    else:
        raise ValueError('unknown selective_range')

    if False in [ True if (position[i]-srange[i][0] >= -tolerance) and (srange[i][1]-position[i] >= tolerance) else False  for i in range(0, 3)]: # be careful on the left boundary
        return False
    else:
        return True

def check_formated_atom(atom):
    """
    check whether the format of atom is valid. 
    Arguments:
        atom: formated atom. if type of coordinate is 'Direct',  you can not specify
            the type. But, for 'Cartesian', must be given. the valid formation:
                ['Na', 0.1, 0.0, 0.0, 'Direct']
                ['Na', 0.1, 0.0, 0.0]
                ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                
                contain species information:
                ['Na1+', 0.1, 0.0, 0.0, 'Direct']
                ['Na1+', 0.1, 0.0, 0.0]
                ['Na1+', 5.234, 0.0, 0.0, 'Cartesian']
    """
    if check_formated_atom_only_direct(atom) or check_formated_atom_only_cartesian(atom):
        return True
    else:
        return False
    
#@profile
def check_formated_atom_only_direct(atom):            
    """
    check whether the format of atom is valid. 
    Arguments:
        atom: formated atom. The type of coordinate is only 'Direct',  you can not specify
            the type. The valid formation:
                ['Na', 0.1, 0.0, 0.0, 'Direct']
                ['Na', 0.1, 0.0, 0.0]
                
                contain species information:
                ['Na1+', 0.1, 0.0, 0.0, 'Direct']
                ['Na1+', 0.1, 0.0, 0.0]
    """    
    if (3 < len(atom) < 6) and (check_symbol(atom[0]) or check_species(atom[0])) and (check_formated_position_only_direct(atom[1:])):
        return True
    else:
        return False

def check_formated_atom_only_cartesian(atom):            
    """
    check whether the format of atom is valid. 
    Arguments:
        atom: formated atom. The type of coordinate is only 'Cartesian',  you must specify
            the type. The valid formation:
                ['Na', 2.3, 1.4, 0.0, 'Cartesian']
                
                contain species information:
                ['Na1+', 2.3, 1.4, 0.0, 'Cartesian']
    """
    if (len(atom) == 5) and (check_symbol(atom[0]) or check_species(atom[0])) and (check_formated_position_only_cartesian(atom[1:])):
        return True
    else:
        return False


def check_formated_position(position):
    """
    check whether the format of position is valid. 
    Arguments:
        position: formated position. if type of coordinate is 'Direct',  you can not specify
            the type. But, for 'Cartesian', must be given. the valid formation:
                [0.1, 0.0, 0.0, 'Direct']
                [0.1, 0.0, 0.0]
                [5.234, 0.0, 0.0, 'Cartesian']
    """
    if check_formated_position_only_direct(position) or check_formated_position_only_cartesian(position):
        return True
    else:
        return False
    

def check_formated_position_only_direct(position):
    """
    check whether the format of position is valid. 
    Arguments:
        position: formated position. The type of coordinate is only 'Direct',  you can not specify
            the type. The valid formation:
                [0.1, 0.0, 0.0, 'Direct']
                [0.1, 0.0, 0.0]
    """
    if ((len(position) == 3) or (len(position) == 4 and position[-1].strip().lower().startswith('d'))) and not(False in [is_int_or_float(x) for x in position[:3]]):
        return True
    else:
        return False

def check_formated_position_only_cartesian(position):
    """
    check whether the format of position is valid.
     
    Arguments:
        position: formated position. The type of coordinate is only 'Cartesian',  you must specify
            the type. The valid formation:
                [5.234, 0.0, 0.0, 'Cartesian']
    """
    if (len(position) == 4 and position[-1].strip().lower().startswith('c') and not(False in [is_int_or_float(x) for x in position[:3]])):
        return True
    else:
        return False

def check_constraint(constraint):
    """
    check whether the value of constraint is boolean.
    
    Arguments:
        constraint: array of boolean. The valid formation: [False, False, True].
    """
    for value in constraint:
        if not isinstance(value, (bool, np.bool_)):
            return False
    return True

        
def check_formated_angle(theta):
    """
    check whether the format of angle is valid. 
    Arguments:
        theta: formated angle. The valid formation:
                [90, 'Degree']
                [1.2, 'Radian']
    """
    if (isinstance(theta, list) or isinstance(theta, np.ndarray)) and len(theta) == 2 and (theta[-1].strip().lower().startswith('d') or theta[-1].strip().lower().startswith('r')):
        return True
    else:
        return False

def check_formated_angle_only_degree(theta):
    """
    check whether the format of angle is valid. 
    Arguments:
        theta: formated angle. The valid formation:
                [90, 'Degree']
    """
    if (isinstance(theta, list) or isinstance(theta, np.ndarray)) and len(theta) == 2 and theta[-1].strip().lower().startswith('d'):
        return True
    else:
        return False

def check_formated_angle_only_radian(theta):
    """
    check whether the format of angle is valid. 
    Arguments:
        theta: formated angle. The valid formation:
                [90, 'Radia']
    """
    
    if (isinstance(theta, list) or isinstance(theta, np.ndarray)) and len(theta) == 2 and theta[-1].strip().lower().startswith('r'):
        return True
    else:
        return False
    
def check_symbol(symbol_of_element):
    """
    check the validity of given symbol.
    
    Arguments:
        symbol_of_element: symbol of element.
    """
    from jamip.structure.atomic_number import number
    
    return True if symbol_of_element in number.keys() else False
    
def check_species(name_of_species):
    """
    check the validity of given spceies.
    
    Arguments:
        name_of_species: name of species. i.e. 'Na+', 'Fe3+'
    """
    import re
    from jamip.structure.atomic_number import number
    
    result = re.match('([A-Z][a-z]?)([0-9.]*)[-+]$', name_of_species)
    if result:
        symbol, valence = result.groups()
        if symbol in number and ( valence == '' or is_int_or_float(valence) ):
            return True

    return False
    
def check_consistency_between_memory_and_db(structure):
    """
    check data consistency between memory and database.
    
    Arguments:
        structure: 
    """
    from ..materials.structure import Structure
    from ..materials.molStructure import MolStructure
    
    # check
    if not(isinstance(structure, Structure) or isinstance(structure, MolStructure)):
        raise ValueError('unknown structure')
    
    print('\n++++++++++ chek data consistency between memory and database ++++++++++\n')
    # structure:
    if isinstance(structure, Structure):
        # entries
        result=None
        entries=structure.entries
        entries_db=structure.entry_set.all()
        if not is_same(entries, entries_db): result=False
#        print('entries:', entries)
        print('structure.entries/entry_set: {}'.format('passed' if result is None else 'failed'))  
    # elements
    result=None
    elements=structure.elements
    elements_db=structure.element_set.all()
    if not is_same(elements, elements_db): result=False
#    print('elements:', elements)
    print('structure.elements/element_set: {}'.format('passed' if result is None else 'failed'))
    if isinstance(structure, Structure):
        # species
        result=None
        species=structure.species
        species_db=structure.species_set.all()
        if not is_same(species, species_db): result=False
#        print('species:', species)
        print('structure.species/species_set: {}'.format('passed' if result is None else 'failed'))
    # atoms
    result=None
    atoms=structure.atoms
    atoms_db=structure.atom_set.all()
    if not is_same(atoms, atoms_db): result=False
#    print('atoms:', atoms)
    print('structure.atoms/atom_set: {}\n'.format('passed' if result is None else 'failed'))
    
    # composition
    if isinstance(structure, Structure):
        # prototypes
        result=None
        prototypes=structure.composition.prototypes
        prototypes_db=structure.composition.prototype_set.all()
        if not is_same(prototypes, prototypes_db): result=False
#        print('structure.composition.prototypes:', prototypes)
        print('structure.composition.prototypes/prototype_set: {}'.format('passed' if result is None else 'failed'))
    # structures
    result=None
    structures=structure.composition.structures
    structures_db=structure.composition.structure_set.all()
    if not is_all_in_collection(list1=structures, collection=structures_db): result=False
#    print('structure.composition.structures:', structures)
    print('structure.composition.structures/structure_set: {}'.format('passed' if result is None else 'failed'))
    # elements
    result=None
    elements=structure.composition.elements
    elements_db=structure.composition.element_set.all()
    if not is_same(elements, elements_db): result=False
#    print('structure.composition.elements:', elements)
    print('structure.composition.elements/element_set: {}\n'.format('passed' if result is None else 'failed'))
    
    # element
    # structures
    result=None
    structure_elements_structures={}
    for element in structure.elements:
        structures=element.structures
        structures_db=element.structure_set.all()
        if not is_all_in_collection(list1=structures, collection=structures_db): result=False
        structure_elements_structures[element]=structures
#    print('structure.elements.structures:', structure_elements_structures)
    print('structure.elements.structures/structure_set: {}'.format('passed' if result is None else 'failed'))
    # compositions
    result=None
    structure_elements_compositions={}
    for element in structure.elements:
        compositions=element.compositions
        compositions_db=element.composition_set.all()
        if not is_all_in_collection(list1=compositions, collection=compositions_db): result=False
        structure_elements_compositions[element]=compositions
#    print('structure.elements.compositions:', structure_elements_compositions)
    print('structure.elements.compositions/composition_set: {}'.format('passed' if result is None else 'failed'))
    if isinstance(structure, Structure):
        # species
        result=None
        structure_elements_species={}
        for element in structure.elements:
            species=element.species
            species_db=element.species_set.all()
            if not is_all_in_collection(list1=species, collection=species_db): result=False
            structure_elements_species[element]=species
#        print('structure.elements.species:', structure_elements_species)
        print('structure.elements.species/species_set: {}\n'.format('passed' if result is None else 'failed'))
    # atoms
    
    if isinstance(structure, Structure):
        # species
        # structures
        result=None
        structure_species_structures={}
        for species in structure.species:
            structures=species.structures
            structures_db=species.structure_set.all()
            if not is_all_in_collection(list1=structures, collection=structures_db): result=False
            structure_species_structures[element]=structures
#        print('structure.species.structures:', structure_species_structures)
        print('structure.species.structures/structure_set: {}\n'.format('passed' if result is None else 'failed'))
        # atoms
    
        # spacegroup
        # structures
        result=None
        structures=structure.spacegroup.structures
        structures_db=structure.spacegroup.structure_set.all()
        if not is_all_in_collection(list1=structures, collection=structures_db): result=False
        structure_species_structures[element]=structures
#        print('structure.spacegroup.structures:', structure_species_structures)
        print('structure.spacegroup.structures/structure_set: {}\n'.format('passed' if result is None else 'failed'))
    
        # prototype
        # structures
        result=None
        if not(structure.prototype is None):
            structures=structure.prototype.structures
            structures_db=structure.prototype.structure_set.all()
            if not is_all_in_collection(list1=structures, collection=structures_db): result=False
            structure_species_structures[element]=structures
#        print('structure.prototype.structures:', structure_species_structures)
        print('structure.prototype.structures/structure_set: {}\n'.format('passed' if result is None else 'failed'))
    
    print('++++++++ end chek data consistency between memory and database ++++++++\n')
    return True if not result else False

    
def is_same(list1, list2):
    """
    check whether two given list arrays are same.
    """
    if (list1 == []) and (list2 == []):
        return True
    elif len(list1) != len(list2):
        return False
    else:
        for v1 in list1:
            if not(v1 in list2): return False
        
    return True

def is_all_in_collection(list1, collection):
    """
    check whether all element in given list array are in collection
    """
    if (list1 == []) and (collection == []):
        return True
    elif len(list1) > len(collection):
        return False
    else:
        for v1 in list1:
            if not(v1 in collection): return False
        
    return True
        

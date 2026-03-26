# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from ..utils.variables import default_constants
import numpy as np
        

class StructureAnalyser(object):
    
    def __init__(self, structure):
        """
        Arguments:
            structure: structure's object.
        """
        self.structure=structure
    
    def niggli_reduce(self, eps=1e-5):
        """
        Niggli reduction.
        
        Arguments:
            eps (default=1e-5): tolerance parameter, but unlike symprec the unit is not a length.
                This is used to check if difference of norms of two basis vectors 
                is close to zero or not and if two basis vectors are orthogonal
                by the value of dot product being close to zero or not. The detail
                is shown at https://atztogo.github.io/niggli/.
        
        Returns:
            Niggli reduction.
        """
        import spglib
        
        cell=self.structure.formatting('cell')
        lattice=cell['lattice']
        niggli_lattice=spglib.niggli_reduce(lattice, eps=eps)
        return niggli_lattice
    
    def delaunay_reduce(self, eps=1e-5):
        """
        Delaunay reduction.
        
        Arguments:
            eps (default=1e-5): tolerance parameter, see niggliReduce.
        """
        import spglib
        
        cell=self.structure.formatting('cell')
        lattice=cell['lattice']
        delaunay_lattice=spglib.delaunay_reduce(lattice, eps=eps)
        return delaunay_lattice
    
    def _calculate_RDF(self, formated_atom, symbol_of_element:list, max_r=10.0, min_r=default_constants.precision.value, dr=0.1):
        """
        radius distribution function (RDF) for atom.
        
        Arguments:
            max_r (default=10.0): max radius (unit: Angstrom).
            min_r (default=1e-3): min radius (unit: Angstrom).
            dr (default=0.1): delta radius (unit: Angstrom).
            symbol_of_element (default=structure.elements)
            
        Returns:
            {r:{symbol:positions}}
        """
        import numpy as np
        import pandas as pd
        #from ..utils.convert import any2cartesian
        
        atom0=self.structure.get_atom(formated_atom=formated_atom) # center
        if atom0 is None:
            raise ValueError('non-existed formated_atom')
        if dr > max_r:
            raise ValueError('beyond the upper boundary (< %.1f): dr' %max_r)
            
        lattice_parameters=self.structure.lattice_parameters
        grange = max_r / np.array(lattice_parameters[:3])
        left_grange = np.floor(atom0.position - grange)   # left of a in lattice for supercell
        right_grange = np.ceil(atom0.position + grange)   # right of a in lattice for supercell
        dim = np.c_[left_grange, right_grange].astype(int)
        grid = np.mgrid[left_grange[0]:right_grange[0]+1,
                        left_grange[1]:right_grange[1]+1,
                        left_grange[2]:right_grange[2]+1].reshape(3,-1).T
        
        # get position by symbol %
        positions = []
        elements = []
        atoms=list(self.structure.atoms)
        for atom1 in atoms:
            if atom1.element.symbol in symbol_of_element:
                positions.append(atom1.position)
                elements.append(atom1.element.symbol)
        positions = np.array(positions)
        elements = np.array(elements)

        # positions [Natom,3] -> supercell_positions [Ngrid, Natom, 3]
        all_positions = (positions[None,:,:] + grid[:,None,:]).reshape(-1,3)
        all_distances = np.linalg.norm( np.dot(all_positions - atom0.position, self.structure.lattice), axis=-1 )
        #formated_distances = np.around( all_distances / dr ) * dr
        df = pd.DataFrame({'distance': all_distances,
        #                   'position': list(np.around(all_positions,8)),
                           'element': np.tile(elements,len(grid))})
        df = df[(df['distance'] > min_r) & (df['distance'] <= max_r)]
        df['formated_distance'] = (np.around( df['distance'] / dr ) * dr)
        return df
    
    def get_RDF_of_atom(self, formated_atom, symbol_of_element=None, max_r=10.0, min_r=default_constants.precision.value, dr=0.1):
        """
        radius distribution function (RDF) for atom.
        
        Arguments:
            max_r (default=10.0): max radius (unit: Angstrom).
            min_r (default=1e-3): min radius (unit: Angstrom).
            dr (default=0.1): delta radius (unit: Angstrom).
            symbol_of_element (default=structure.elements)
            
        Returns:
            list-type [distance, number_of_atoms, [formated_atom1, formated_atom2,...],
                       ...
                      ]
        """
        if symbol_of_element == None:
            symbol_of_element = np.unique([atom.element.symbol for atom in self.structure.atoms])
            
        full_rdf=self._calculate_RDF(formated_atom=formated_atom, symbol_of_element=symbol_of_element, max_r=max_r, min_r=min_r, dr=dr)
        rdf=[] # [distance, number_of_atoms]}
        for name,group in full_rdf.groupby(['formated_distance','element']):
            distance, element = name
            rdf.append([distance, element, len(group)])
        return rdf
    
    def _calculate_RDF_of_all(self, max_r=10.0, min_r=default_constants.precision.value, dr=0.1):
        """
        get the radial distribution function (RDF) of all atom.
        """
        """
        radius distribution function (RDF) for atom.
        
        Arguments:
            max_r (default=10.0): max radius (unit: Angstrom).
            min_r (default=1e-3): min radius (unit: Angstrom).
            dr (default=0.1): delta radius (unit: Angstrom).

        Returns:
            {r:{symbol:positions}}
        """
        #import math
        import numpy as np
        import pandas as pd
        #from collections import OrderedDict
        #from ..utils.convert import any2cartesian
        
        if dr > max_r:
            raise ValueError('beyond the upper boundary (< %.1f): dr' %max_r)

        # get all position %
        positions = []
        elements = []
        atoms=list(self.structure.atoms)
        for atom1 in atoms:
            positions.append(atom1.position)
            elements.append(atom1.element.symbol)
        positions = np.array(positions)
        elements = np.array(elements)

        # get range %
        grange = max_r / np.array(self.structure.lattice_parameters[:3])
        left_grange = np.floor(np.min(positions) - grange)   # left of a in lattice for supercell
        right_grange = np.ceil(np.max(positions) + grange)   # right of a in lattice for supercell
        dim = np.c_[left_grange, right_grange].astype(int)
        grid = np.mgrid[left_grange[0]:right_grange[0]+1,
                        left_grange[1]:right_grange[1]+1,
                        left_grange[2]:right_grange[2]+1].reshape(3,-1).T
        X,Y = np.meshgrid(elements,elements)
        element_tuples = np.c_[X.ravel(),Y.ravel()]
        all_element_tuples = np.tile(element_tuples,(len(grid),1))
        # positions [Natom,3] -> vectors [Natom, Natom, 3]
        vectors = (positions[None,:,:] + positions[:,None,:]).reshape(-1,3)
        # vectors -> supercell_vectors [Ngrid, Natom, Natom, 3]
        all_vectors = (vectors[None,:,:] + grid[:,None,:]).reshape(-1,3)
        all_distances = np.linalg.norm( np.dot(all_vectors, self.structure.lattice), axis=-1 )
        df = pd.DataFrame({'distance': all_distances,
        #                   'position': map(tuple,np.around(all_positions,8)),
                           'element': list(all_element_tuples)})
        df = df[(df['distance'] > min_r) & (df['distance'] <= max_r)]
        df['element'] = df['element'].apply(tuple)
        #list2str = lambda x: '-'.join(x)
        #df['element'] = df['element'].apply(list2str)
        df['formated_distance'] = (np.around( df['distance'] / dr ) * dr)
        return df
    
    def get_RDF_of_all(self, max_r=10.0, min_r=default_constants.precision.value, dr=0.1):
        """
        radius distribution function (RDF) for all atom.
        
        Arguments:
            max_r (default=10.0): max radius (unit: Angstrom).
            min_r (default=1e-3): min radius (unit: Angstrom).
            dr (default=0.1): delta radius (unit: Angstrom).
            
        Returns:
            list-type [distance, number_of_atoms, [formated_atom1, formated_atom2,...],
                       ...
                      ]
        """
        full_rdf=self._calculate_RDF_of_all(max_r=max_r, min_r=min_r, dr=dr)
        rdf=[] # [distance, number_of_atoms]}
        for name,group in full_rdf.groupby(['formated_distance','element']):
        #for name,group in full_rdf.groupby(['formated_distance']):
            distance,element = name
            rdf.append([distance, element, len(group)])
        return rdf
    
    def get_coordination_atoms_of_atom(self, formated_atom, cutoff=3.0):
        """
        get the coordination atoms of given atom.
        
        Arguments:
            cutoff (default=3.0): cutoff radius when calculating the coordiantion atoms.
            
        Returns:
            the coordination atoms of given atom. list-type [formated_atom1, formated_atom2,...]
        """        
        raise RuntimeError("lost effectiveness")
        full_rdf=self.get_RDF_of_atom(formated_atom=formated_atom, max_r=cutoff)
        coordination_atoms=[]
        for v in full_rdf:
            if v[0] <= cutoff:
                coordination_atoms += v[2]
        coordination_atoms=sorted(coordination_atoms, key=lambda v:v[0])
        return coordination_atoms
    
    def get_coordination_number_of_atom(self, formated_atom, cutoff=3.0):
        """
        get the coordination number of given atom within the cutoff radius.
        """
        full_rdf=self.get_RDF_of_atom(formated_atom=formated_atom, max_r=cutoff)
        coordination_numbers = 0
        for v in full_rdf:
            if v[0] <= cutoff:
                coordination_numbers += v[2]
        return coordination_numbers
    
    def get_averaged_coordination_number(self, symbol_of_element, cutoff=3.0):
        """
        get the averaged coordination number of given element.
        """
        # check
        atoms=self.structure.get_atoms_of_element(symbol=symbol_of_element)
        averaged_coordination_number=0.0
        for atom in atoms:
            averaged_coordination_number += self.get_coordination_number_of_atom(formated_atom=atom.to_formated_atom(), cutoff=cutoff)
        averaged_coordination_number /= len(atoms)
        return averaged_coordination_number    
    
    def get_atomic_packing_factor(self, atomic_radii=None):
        """
        get the atomic packing factor.
        """
        atom_volumes = 0
        for atom in self.structure.atoms:
            if atom.element.atomic_radius == None:
                atom.element.supplement()
            radius = atom.element.atomic_radius/100  # pm -> Ang
            atom_volumes += 4/3 * np.pi * radius**3
        return atom_volumes / self.structure.volume
    
    def get_tolerance_factor(self, r_a, r_b, r0):
        """
        get the Goldschmidt tolerance factor by given atomic radii.
        https://blog.csdn.net/weixin_44210796/article/details/106120423
        """
        return (r_a + r0) / (np.sqrt(2) * (r_b + r0))
    
    def get_XRD_pattern(self, target_of_anode_material=''):
        """
        calculate XRD pattern.
        
        Arguments:
            target_of_anode_material: target of anode material. 
                Anode   Cr   Fe   Co   Cu   Mo   Ag
                K_alpha 2.29 1.94 1.79 1.54 0.71 0.56
        
        Return:
            XRD pattern.
        """
        pass
    

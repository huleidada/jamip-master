# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from .atom import Composition

class Atom:
    def __init__(self, specie, coord):
        self.specie = specie
        self.coord = coord

class Molecule:
    """
    define a crystal structure or molecule, including,
    
    Attributes:
        species_of_elements: list, sepcies of elements;
        number_of_atoms:     list, numbers of species;
        atomic_positions:    list, atomic position defined by Atom Object;
        comment_line:        string, introduction of structure;
        scale_factor:        float, default, 1.0;
        lattice:             numpy.array(3x3), default, None; 
    """

    magnetic = None

    def __init__(self, *args, **kwargs):

        self._comment = 'Created by JAMIP'
        self.__positions = None
        self.__number = None
        self.__species = None
        self.__cell = None
        self.__connections = []
        self.__all_connections = defaultdict(list)
	
    @property
    def comment_line(self):
        return self._comment 

    @comment_line.setter
    def comment_line(self, value=None):
        self._comment = value

    @property
    def species_of_elements(self):
        return self.__species

    @species_of_elements.setter
    def species_of_elements(self, value=None):
        self.__species = value

    @property
    def number_of_atoms(self):
        return self.__number

    @number_of_atoms.setter
    def number_of_atoms(self, value=None):
        self.__number = value

    @property
    def lattice(self):
        if self.__cell is not None:
            return self.__cell.vectors

    @lattice.setter
    def lattice(self, value):
        from .atom import Cell
        self.__cell = Cell(value)

    @property
    def lattice_parameters(self):
        """
        Return:
            lattice parameters [a, b, c, alpha, beta, gamma].
        """
        if self.__cell is not None:
            return self.__cell.parameters

    @property
    def connections(self):
        return self.__connections

    @connections.setter
    def connections(self, value):
        """
        save infomation for mol/sdf
        """
        from copy import deepcopy
        indices = sum([v[:2] for v in value],[])
        assert max(indices) < len(self.positions), "Invalid mol connections indices."
        assert set([len(v) for v in value]) == {3}, "Invalid mol connections shape."
        self.__connections = deepcopy(value)

    def get_connections(self, type='gaussian'):
        
        from collections import defaultdict

        # array to dict
        if type == 'gaussian':
            if self.connections is None: return None
            data = {i:defaultdict(float) for i in range(len(self.positions))}
            for i,j,k in self.connections:
                data[i][j] += k 
            return data
        elif type == "pdb":
            data = defaultdict(list)
            for i,j in self.all_connections['R']:
                data[i].append(j)
                data[j].append(i)
            return data
        else:
            raise ValueError("Unsupport data type.")

    @property
    def all_connections(self):
        return self.__all_connections

    @all_connections.setter
    def all_connections(self, value):
        """
        save infomation for pdb
        """
        acs = defaultdict(list)
        for row in value:
            if len(row) == 2: 
                acs['R'].append(row)
            elif len(row) == 3: 
                acs['A'].append(row)
            elif len(row) == 4: 
                acs['D'].append(row)
            elif len(row) == 5: 
                # TODO what's this?
                pass
            else:
                raise ValueError(f"Invalid connections: {row}")
        self.__all_connections = acs

    @property
    def positions(self):
        return np.array([atom.coord for atom in self.__positions])

    @property
    def atomic_positions(self):
        return self.__positions

    @atomic_positions.setter
    def atomic_positions(self, value=None):
        elements = self.get_elements(type='symbol')
        assert len(elements) == len(value), f"Nelement = {len(elements)}, but Ncoords = {len(value)}"
        positions =  []
        for i,atom in enumerate(value):
            positions.append(Atom(elements[i], atom))

        self.__positions = positions

    @classmethod
    def elements2species(cls, elements, positions=None, sort=False):
        """ Convert elements list to species groups & sort positions by elements

        Args:
            elements (list/array): List of atomic numbers (int) or element symbols (str)
            positions (list/array): List of atomic positions corresponding to elements
            sort (bool, optional): Whether to sort elements by atomic number. 
                                If False, preserves original order of first occurrence.
                                Defaults to True.

        Returns:
            tuple: Contains three arrays:
                - unique_elements: Array of unique element symbols
                - counts: Counts of each element type
                - grouped_positions: List of position arrays grouped by element type
        """
        from .atomic_number import atomic

        if positions is None:
            active = None
            species = []
            numbers = []
            for i in elements:
                if i != active:
                    active = i
                    species.append(i)
                    numbers.append(1)
                else:
                    numbers[-1] += 1
            species = [atomic[v] if isinstance(v, (int, np.integer)) else v for v in species]
            return np.array(species), np.array(numbers)

        else:
            # Process unique elements
            values, ids, indices, counts = np.unique(elements, return_index=True, 
                                                     return_inverse=True, return_counts=True)
            unique_elements = np.array([atomic[v] if isinstance(v, (int, np.integer)) else v 
                                        for v in values])
            
            # Handle sorting if disabled
            if not sort:
                sort_order = np.argsort(ids)
                unique_elements = unique_elements[sort_order]
                counts = counts[sort_order]
                indices = np.argsort(sort_order)[indices]
         
            # Group positions by element
            positions = np.array(positions)
            grouped_positions = [positions[indices == i] for i in range(len(unique_elements))]
            grouped_positions = np.concatenate(grouped_positions, axis=0)
            
            return unique_elements, counts, grouped_positions

    def __len__(self):
        return sum(self.__number)

    @property
    def composition(self):
        """
        Returns:
            int: The formula units Z of the structure.
        """
        return Composition.from_elements(self.get_elements(type='symbol'))

    def to_cell(self):
        #return (self.__atomic_coord, self.get_elements())
        return (self.get_positions(), self.get_elements())

    @classmethod
    def from_cell(cls, cell, comment='jamip', standard=False):
        assert len(cell) == 2
        positions, elements = cell
        obj = cls()
        obj.comment_line = comment
        if standard is True:
            species,numbers,positions = cls.elements2species(elements, positions)
            obj.species_of_elements = species
            obj.number_of_atoms = numbers
            obj.atomic_positions = positions
        else:
            species, numbers = obj.elements2species(elements)
            obj.species_of_elements = species
            obj.number_of_atoms = numbers
            obj.atomic_positions = positions
        return obj

    def get_formula(self, reduced=False, split='', sort=False):
        """ Get the formula of the structure.
        
        Args:
            reduced (bool): Whether to divide the number of atoms by the formula units Z. Defaults to False.
            split (str): The split character. Defaults to ''.
            sort (bool): Whether to sort the positions by element. Defaults to True.

        Returns:
            str: The formula of the structure.
        """
        div = 1
        if reduced is True:
            div = np.gcd.reduce(self.__number)

        if sort is True:
            values,indices = np.unique(self.__species, return_index=True)
        else:
            indices = range(len(self.__number)) 

        formula = ''
        for i,e in enumerate(indices):
            e = self.__species[i]
            n = self.__number[i]
            if n/div == 1 and not sort:
                formula += '%s%s' %(e,split)
            else:
                formula += '%s%d%s' %(e,n/div,split)
        return formula.rstrip(split)
        
    def get_elements(self, type='number', safe=False):
        """Get atomic elements in specified format.
        
        Args:
            type (str): Output format specification:
                - 'number': returns atomic numbers (default)
                - 'symbol': returns element symbols
        
        Returns:
            numpy.ndarray: Array of atomic numbers or element symbols, 
            repeated according to atom counts
            
        Example:
            >>> struct.get_elements('symbol')
            ['H', 'H', 'O']  # For water molecule
        """
        from .atomic_number import number
        if not safe:
            if type == 'symbol':
                species = self.__species
            elif type == 'number':
                species = [number[i] for i in self.__species]
            return np.repeat(species, self.__number)
        else:
            if type == 'symbol':
                return np.array([atom.specie for atom in self.__positions])
            elif type == 'number':
                return np.array([number[atom.specie] for atom in self.__positions])

    def get_positions(self, type='cartesian'):
        """Get atomic positions in specified coordinate system.
        
        Args:
            type (str): Coordinate system type:
                - 'cartesian': Cartesian coordinates (default)
                - 'direct': Fractional coordinates (requires lattice)
        
        Returns:
            numpy.ndarray: Nx3 array of atomic positions
        
        Raises:
            ValueError: If requesting fractional coordinates without lattice
        
        Note:
            Fractional coordinates are calculated as r_frac = r_cart · cell^(-1)
        """
        if type == 'cartesian':
            return self.positions
        elif type == 'direct':
            if self.__cell is None:
                raise ValueError('Molecular system requires lattice for fractional coordinates')
            else:
                return self.positions @ np.linalg.inv(self.lattice)

    def get_all_distances(self):
        """Calculate pairwise distances between all atoms.
        
        Returns:
            numpy.ndarray: Symmetric NxN matrix of interatomic distances (in Angstroms)
            Diagonal elements are zero (self-distances excluded)
        
        Note:
            Distance matrix is computed as d_ij = ||r_i - r_j||
            For periodic systems, use get_all_distances_pbc() instead
        """
        positions = self.positions
        vectors = positions[None,::] - positions[:,None,:] 
        distances = np.linalg.norm(vectors)
        return distances

    def get_structure(self, vaccum=20, center=True):
        """Convert to Structure object with optional vacuum padding and centering.
        
        Args:
            vaccum (float): Vacuum padding size (in Angstroms) when no cell exists (default: 20)
            center (bool): Whether to center positions in the cell (default: True)
        
        Returns:
            Structure: New structure object with specified configuration
        
        Note:
            For molecular systems (no cell), creates cubic box with given vacuum size
            Centering places atoms at cell center by adding half of lattice vectors
        """
        from .structure import Structure 
        obj = Structure()
        if self.__cell is None:
            obj.lattice = np.eye(3) * vaccum
        else:
            obj.lattice = self.lattice
        obj.comment_line = self.comment_line
        obj.atomic_coord_format = 'cartesian'
        obj.species_of_elements = self.species_of_elements
        obj.number_of_atoms = self.number_of_atoms 
        if center:
            obj.atomic_positions = self.positions + np.sum(obj.lattice, axis=0) * 0.5
        else:
            obj.atomic_positions = self.positions 
        return obj

    def add_atom(self, atom):
        """Add an atom to the structure.

        Args:
            atom (Atom): Atom object to be added.
        """
        assert isinstance(atom, Atom) == True
        self.__positions.append(atom)

    def get_atom(self, position, specie=None, tol=1e-3, is_exist=True):
        """Get the index of the atom closest to the given position.

        Args:
            position (list): Position of the atom to be found.
            specie (str, optional): Element symbol of the atom to be found. Defaults to None.
            tol (float, optional): Tolerance for finding the atom. Defaults to 1e-3.
            is_exist (bool, optional): Whether to raise an error if the atom is not found. Defaults to True.

        Returns:
            list: List of indices of the atoms closest to the given position.
        """
        if isinstance(position, Atom):
            position = position.coord
        positions = self.get_positions(type='cartesian')
        vectors = positions - position
        distances = np.linalg.norm(vectors, axis=1)
        assert len(distances) == len(positions) #! check
        if specie is None:
            indices = np.where(distances < tol)[0]
        else:
            elements = self.get_elements(type='symbol', safe=True)
            indices = np.where((distances < tol) & (elements == specie))[0]
        if len(indices) == 0:
            if is_exist: 
                raise RuntimeError("not find atom.")
        return indices

    def del_atom(self, position, specie=None, **kwargs):
        """ Delete an atom from the structure.

        Args:
            position (list): Position of the atom to be deleted.
            specie (str, optional): Element symbol of the atom to be deleted. Defaults to None.
            **kwargs: Additional arguments for get_atom method.
        """
        indices = self.get_atom(position, specie, **kwargs)
        if len(indices):
            atoms = []
            for i,atom in enumerate(self.__positions):
                if i not in indices:
                    atoms.append(atom)
            self.__positions = atoms

    def substitute_atom(self, position, specie, **kwargs): 
        """ Substitute an atom in the structure.

        Args:
            position (list): Position of the atom to be substituted.
            specie (str): Element symbol of the new atom.
            **kwargs: Additional arguments for get_atom method.
        """
        indices = self.get_atom(position, specie=None, **kwargs)
        if len(indices):
            atoms = []
            for i,atom in enumerate(self.__positions):
                if i in indices:
                    atom.specie = specie

    def update(self, standard=True):
        """
        Update the structure by recalculating the atomic positions and species.

        This method updates the structure's atomic positions and species based on the current state of the structure.
        It ensures that the atomic positions are sorted according to the species and updates the internal state of the structure.
        """
        elements = self.get_elements(type='symbol', safe=True)
        positions = self.get_positions()
        if standard is True:
            species,numbers,positions = self.elements2species(elements, positions, sort=False)
            self.species_of_elements = species
            self.number_of_atoms = numbers
            self.atomic_positions = positions
        else:
            species, numbers = self.elements2species(elements)
            self.species_of_elements = species
            self.number_of_atoms = numbers
            self.atomic_positions = positions

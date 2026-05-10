import numpy as np
from typing import NamedTuple
# from jamip.utils.utils import lazy_property
from functools import cached_property
from .atomic_number import number

__all__ = ['Cell', 'SelectDynamic', 'Atom', 'Composition']
__contributor__ = 'xingang zhao, xingang.zhao@colorado.edu'


class SelectDynamic(NamedTuple):
    """
    cls define the frozen of the atoms;
        Default:: not fronze x, y, z;
    """
    x:bool = True
    y:bool = True
    z:bool = True

    @property
    def xyz(self):
        return np.where(np.array([self.x,self.y,self.z]),'T','F')
   
    def __repr__(self):
        return "[{0[0]}, {0[1]}, {0[2]}]".format(self.xyz) 

class Cell(object):

    """
    cls Lattice aim to define the lattice vector. 
   
    properties:
        length a, b, c; 
        angle alpha, beta, gamma; 
        cell volume; 
        scale: scale lattice vectors;
        vectors: 3x3 numpy.array; 
   function:
        get_cell: return self.vectors, i.e., 3x3 vectors;	 
    """

    def __init__(self, cell, scale=1.0):
        
        self.__cell = np.array(cell, dtype=float) * scale 
        self.__scale = scale
   
    # @lazy_property
    @cached_property
    def a(self):
        return np.linalg.norm(self.__cell[0])

    # @lazy_property
    @cached_property
    def b(self):
        return np.linalg.norm(self.__cell[1])

    # @lazy_property
    @cached_property
    def c(self):
        return np.linalg.norm(self.__cell[2])

    # @lazy_property
    @cached_property
    def alpha(self):
        value=np.dot(self.__cell[1],self.__cell[2])/(self.b*self.c)
        return np.arccos(value)/np.pi*180

    # @lazy_property
    @cached_property
    def beta(self):
        value=np.dot(self.__cell[0],self.__cell[2])/(self.a*self.c)
        return np.arccos(value)/np.pi*180

    # @lazy_property
    @cached_property
    def gamma(self):
        value=np.dot(self.__cell[0],self.__cell[1])/(self.a*self.b)
        return np.arccos(value)/np.pi*180
    
    # @lazy_property
    @cached_property
    def volume(self):
        #return np.abs(np.dot(np.cross(self.__cell[0],self.__cell[1]),self.__cell[2]))
        return abs(np.linalg.det(self.__cell))

    @property     
    def vectors(self):
        return np.array(self.__cell, dtype=float)

    @property
    def reciprocal(self):
        return np.linalg.inv(self.__cell)

    @property 
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value:float=1.0):
        self.__scale = value
        self.__cell = self.__cell*value

    def get_spacegroup(self, symprec=1e-1):
        import spglib 
        cell = (self.__cell, [[0,0,0]], [0])
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        return dataset['number'] 

    @property
    def parameters(self):
        """
        Return:
            lattice parameters [a, b, c, alpha, beta, gamma].
        """
        va = self.__cell[0]
        vb = self.__cell[1]
        vc = self.__cell[2]

        a=np.linalg.norm(va)
        b=np.linalg.norm(vb)
        c=np.linalg.norm(vc)
        alpha=np.degrees(np.arccos(np.clip(np.dot(vb/b, vc/c), -1, 1)))
        beta =np.degrees(np.arccos(np.clip(np.dot(va/a, vc/c), -1, 1)))
        gamma=np.degrees(np.arccos(np.clip(np.dot(va/a, vb/b), -1, 1)))
        return np.array([a,b,c,alpha,beta,gamma])

    @classmethod
    def from_parameters(cls, a:float, b:float, c:float, alpha:float, beta:float, gamma:float, reduce_method=None, eps=1e-5):
        import spglib 

        angles_r = np.radians([alpha, beta, gamma])
        cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
        sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)
        
        val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
        # Sometimes rounding errors result in values slightly > 1.
        gamma_star = np.arccos(np.clip(val, -1, 1))

        vector_a = [a * sin_beta, 0.0, a * cos_beta]
        vector_b = [
            -b * sin_alpha * np.cos(gamma_star),
             b * sin_alpha * np.sin(gamma_star),
             b * cos_alpha,
        ]
        vector_c = [0.0, 0.0, float(c)]
        lattice = np.array([vector_a, vector_b, vector_c])

        # reduce %
        if reduce_method == 'niggli':
            lattice = spglib.niggli_reduce(lattice, eps=eps)
        elif reduce_method == 'delaunay':
            lattice = spglib.delaunay_reduce(lattice, eps=eps)

        return cls(lattice)

    def __repr__(self):
        s=''
        if self.__cell is not None:
            s = "lattice info:\n"
            s += " length    a (A) : %f\n" % (self.a)  
            s += " length    b (A) : %f\n" % (self.b)  
            s += " length    c (A) : %f\n" % (self.c)  
            s += " cell volume(A^3): %f\n" % (self.volume)  
            s += " lattice vectors:\n"
            s += ' '.join("%7.4f" % (v) for v in self.__cell[0])+'\n'
            s += ' '.join("%7.4f" % (v) for v in self.__cell[1])+'\n'
            s += ' '.join("%7.4f" % (v) for v in self.__cell[2])+'\n'

        return s
    
class Atom(object):
    """
    cls to define one position, including,
          element
          occupied coordination 
          charge 
          magnetic
          constraint
    """ 
    def __init__(self, element=None, position=None, cell=None, charge=None, magnetic=None,\
                 freeze=None, velocity=None, direct:bool=True, **kwargs):
	
        self.__specie = element
        self.__cell = cell
        self.__charge = charge
        self.__velocity = velocity
        self.__magnetic = magnetic 
        self.freeze = freeze 
        if direct == True:
            self.scale_coord = position
        else:
            self.coord = position

    @property
    def lattice(self):
        return self.__cell.vectors

    @lattice.setter
    def lattice(self,value):
        if isinstance(value, Cell):
            self.__cell = value
        else:
            self.__cell = Cell(value)

    @property
    def specie(self):
        return self.__specie

    @specie.setter
    def specie(self, value:str):
        self.__specie = value 

    @property
    def atomic_number(self):
        return number[self.__specie]

    @property
    def coord(self):
        if self.__coord is None:
            self.__coord = np.dot(self.__scale_coord, self.__cell.vectors)
        return self.__coord

    @coord.setter
    def coord(self, value):
        self.__coord = np.array(value, dtype=float)
        self.__scale_coord = None

    @property
    def scale_coord(self):
        if self.__scale_coord is None:
            self.__scale_coord = np.dot(self.__coord, self.__cell.reciprocal)
        return self.__scale_coord

    @scale_coord.setter
    def scale_coord(self, value):
        self.__coord = None
        self.__scale_coord = np.array(value, dtype=float)

    @property 
    def magmom(self):
        return self.__magnetic

    @magmom.setter
    def magmom(self, value):
        self.__magnetic = np.array(value).reshape(-1)  

    @property 
    def charge(self):
        return self.__charge 
    
    @charge.setter
    def charge(self, value):
        if not (value is None):
            self.__charge = float(value)
      
    @property  
    def freeze(self):
        return SelectDynamic(*self.__freeze)
    
    @freeze.setter
    def freeze(self, value):
        from jamip.utils.convert import format_bool

        if isinstance(value, (bool,str)):
            value = [format_bool(value)] * 3
        elif value is None:
            value = []
        elif len(value) == 3:
            value = [format_bool(i) for i in value]
        else:
            raise ValueError('Atom freeze set error.')
        self.__freeze = value
 
    @property  
    def velocity(self):
        return np.array(self.__velocity, dtype=float)
    
    @velocity.setter
    def velocity(self, value):
        self.__velocity = value

    @property
    def elementinfo(self):
        from .elementInfo import Element
        return Element.from_symbol(self.specie)

    def __repr__(self):
        s = "(Element: %s, Position: %s" % (self.specie, self.coord)
        for key,value in self.properties.items():
            s += f", {key.capitalize()}: {value}"
        s += ")"   
        return s 
   
    @property
    def properties(self):
        props = {}
        for key in ['freeze', 'velocity', 'charge', 'magmom']:
            value = getattr(self, key)
            if not (value is None):
                props[key] = value
        return props
        
        
class Composition(object):
    def __init__(self, species, numbers):
        self.__species = species
        self.__numbers = numbers

    @property
    def numbers(self):
        return self.__numbers

    @property
    def species(self):
        return self.__species

    def get_formula(self, sort=True, reduced=False, split=''):
        div = 1
        if reduced is True:
            div = self.Z

        if sort is True:
            elements = np.repeat(self.__species, self.__numbers) 
            species, numbers = np.unique(elements, return_counts=True)
        else:
            species = self.__species
            numbers = self.__numbers

        formula = ''
        for e,n in zip(species, numbers):
            if n/div == 1 and not sort:
                formula += '%s%s' %(e,split)
            else:
                formula += '%s%d%s' %(e,n/div,split)

        return formula.rstrip(split)        

    @property
    def formula(self):
        return self.get_formula(sort=True, reduced=False, split='')

    @property
    def reduced_formula(self):
        return self.get_formula(sort=True, reduced=True, split='')

    def get_es_formula(self, reduced=True, split=''):
        "Electronegativity Series"
        from .elementInfo import Element
        div = 1
        if reduced is True:
            div = self.Z

        xs = [Element.from_symbol(s).X for s in self.__species] 
        indices = np.argsort(xs)

        formula = ''
        for i in indices:
            e = self.__species[i]
            n = self.__numbers[i]
            if n/div == 1 and split == '':
                formula += '%s%s' %(e,split)
            else:
                formula += '%s%d%s' %(e,n/div,split)
        return formula.rstrip(split)        

    @property
    def ABformula(self):
        Z = self.Z
        general_formula = ''
        for symbol,num in zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',np.sort(self.__numbers)):
            general_formula += '%s%d' %(symbol, num/Z)
        return general_formula

    @property
    def chemsys(self):
        species = np.unique(self.__species).tolist()
        return '-'.join(species)

    @property
    def Z(self):
        return np.gcd.reduce(self.__numbers)
    
    @property
    def best_valence(self):        
        from .valence import get_best_valence
        return get_best_valence(self.__species, self.__numbers)
    
    @classmethod
    def from_dict(cls, d):
        species = []
        numbers = []
        for k,v in d.items():
            if v==0: continue
            species.append(k)
            numbers.append(v)
        return Composition(species, numbers)

    @classmethod
    def from_formula(cls, formula):
        import re
        species = []
        numbers = []
        for e,n in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
            species.append(e)
            if n == '': n = 1
            else: n = int(n)
            numbers.append(n)
        return Composition(species, numbers)
    
    @classmethod
    def from_elements(cls, elements):
        from collections import defaultdict
        d = defaultdict(int)
        for e in elements:
            d[e] += 1
        return cls.from_dict(d)
    
    def as_dict(self):
        """List of atom positions.

        Returns:
            list: list of atom positions.
        """
        from collections import defaultdict
        value = defaultdict(int)
        for i,j in zip(self.__species,self.__numbers):
            value[i] += j
        return value
    
    def to_elements(self):
        elements = []
        for specie, number in zip(self.__species, self.__numbers):
            elements += [specie]*number
        return elements

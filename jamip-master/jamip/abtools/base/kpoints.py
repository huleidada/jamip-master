import numpy as np
from typing import NamedTuple
from collections.abc import Sequence

class Kpath:

    def __init__(self, *args, **kwargs):
        pass

    def __setattr__(self, key, item):
        if isinstance(item, int):
            self.__dict__[key] = item
        elif isinstance(item, tuple):
            self.__dict__[key] = Kpoints(*item)
        elif isinstance(item, (BandPath, Kpoints)):
            self.__dict__[key] = Kpoints(item)
        else:
            raise ValueError("kpath must be int or tuple or class!")

    def __contains__(self, key):
        return key in self.__dict__

    def __setitem__(self, key, item): 
        self.__dict__[key] = item

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        raise KeyError(key)

    def __deepcopy__(self,memo):
        from copy import deepcopy
        cp = Kpath()
        for key, value in self.__dict__.items():
            cp[key] = deepcopy(value)
        return cp

class KPT(NamedTuple):
    x: float
    y: float
    z: float
    symbol: str=''
    # num: int = None

    @property
    def position(self):
        return np.array([self.x,self.y,self.z], dtype=float)
    
    def __repr__(self):
        return '\n{0[0]:14.8f}{0[1]:14.8f}{0[2]:14.8f} ! {1}'.format(self.position,self.symbol)

    def __str__(self):
        return '{0[0]:.4f} {0[1]:.4f} {0[2]:.4f} ! {1}'.format(self.position,self.symbol)    


class BandPath(Sequence):
    model = 'Line Model'
    ''' class of hym Kpath '''
    def __init__(self, coords, symbols=None, numbers=20, masks=None): 

        if symbols is None:
            symbols = ['']*len(coords)
        else:
            assert len(coords) == len(symbols), "coords number != symbols number in bandpath setting!"

        sites = []
        for i,j in zip(coords, symbols):
            x,y,z = i
            sites.append(KPT(x,y,z,j))

        if isinstance(numbers, int):
            if masks is None:
                masks = [True] * (len(coords)-1) + [False]
            masks = np.array(masks, dtype=bool)
            tmp = np.ones(len(coords), dtype=int)
            tmp[masks] = numbers
            numbers = tmp
        else:
            assert len(numbers) == len(coords), "coords number != numbers number in bandpath setting!"
            numbers = np.array(numbers, dtype=int)
            masks = np.where(numbers>1, True, False)

        self._sites = sites
        self._masks = masks
        self._numbers = numbers

    @property
    def sites(self):
        return self._sites
    
    @property
    def numbers(self):
        return self._numbers
    
    @property
    def num_sites(self) -> int:
        return len(self)
    
    @property
    def value(self):
        return self
    
    @property
    def masks(self):
        return self._masks

    def __contains__(self, site):
        return site in self.sites

    def __iter__(self):
        return self.sites.__iter__()

    def __getitem__(self, ind):
        return self.sites[ind]

    def __len__(self):
        return len(self.sites)

    def set_insert(self, value:int):
        numbers = np.ones(len(self), dtype=int)
        numbers[self.masks] = value
        self._numbers = numbers

    def get_insert(self):
        return int(max(self.numbers))

    def __repr__(self):
        result = ''
        for i in range(len(self)):
            if self.masks[i]:
                result += repr(self.sites[i])
                result += repr(self.sites[i+1])
                result += '\n'
        return result

    def __str__(self):
        paths = []
        for i in range(len(self)):
            if self.masks[i]:
                site1 = self.sites[i]
                site2 = self.sites[i+1]
                key = site1.symbol.strip('\\') + '-' + site2.symbol.strip('\\')
                paths.append(key)
        return 'BandPath(paths="%s", insert=%d)' %('|'.join(paths), self.get_insert())

    @property
    def qeformat(self):
        result = ''
        for i,site in enumerate(self.sites):
            result += '{0[0]:10.6f}{0[1]:10.6f}{0[2]:10.6f}{1:4} ! {2}\n'.format(\
                        site.position, self.numbers[i], site.symbol)
        return result

    @classmethod
    def from_symbols(cls,bandpath:Sequence,kpoints:dict,numbers=20):

        masks = []
        coords = []
        symbols = []
        if isinstance(bandpath[0], (list, tuple)):
            for path in bandpath:
                for i,symbol in enumerate(path):
                    coords.append(kpoints[symbol])
                    symbols.append(symbol)
                    mask = True if i+1 < len(path) else False
                    masks.append(mask) 
        else:
            for i,symbol in enumerate(bandpath):
                coords.append(kpoints[symbol])
                symbols.append(symbol)
                mask = True if i+1 < len(bandpath) else False
                masks.append(mask)

        return cls(coords=coords, symbols=symbols, numbers=numbers, masks=masks)

    def split(self):
        kpaths = {}
        for i in range(len(self)):
            if self.masks[i]:
                site1 = self.sites[i]
                site2 = self.sites[i+1]
                key = site1.symbol.strip('\\') + '-' + site2.symbol.strip('\\')
                value = BandPath(coords=[site1.position, site2.position], 
                                 symbols=[site1.symbol, site2.symbol],
                                 numbers=[self.numbers[i],1],
                                 masks=[True,False])
                kpaths[key] = value
        return kpaths

    def set_mesh(self, lattice, mesh=0.01):

        numbers = []
        for i in range(len(self)):
            if self.masks[i]:
                site1 = self.sites[i]
                site2 = self.sites[i+1]
                step = np.subtract(site2.position,site1.position)
                step_num = int(np.ceil(np.linalg.norm(np.dot(np.linalg.inv(lattice),step))/mesh))
                numbers.append(step_num)
            else:
                numbers.append(1)                
        self._numbers = numbers

class Kpoints(object):

    """
    Aim to produce the kmesh
    """
    _model_ = None
    _value_ = None
    _comment_ = ''
 
    def __init__(self, *args, **kwargs):

        model = None
        number = 30
        if len(args) == 1:
            value, = args
        elif len(args) == 2:
            model, value = args
        elif len(args) == 3:
            model, value, number = args
        else:
            raise ValueError("Kpoints args greater than 3!")

        if model is None:

            if isinstance(value, float):
                self._model_ = 'kspacing'
                self._value_ = value

            elif isinstance(value, Kpoints):
                self._model_ = value.model
                self._value_ = value.value
                self._comment_ = value.comment

            elif isinstance(value, BandPath):
                self._model_ = 'Line Model'
                self._value_ = value

            else:
                raise ValueError("Invalid KPOINTS type.")

        elif isinstance(model, str):

            model = model.lower()[0]
 
            if model == 'a':
                self._model_ = 'Auto'
                self._value_ = int(value)

            if model == 'k':
                self._model_ = 'kspacing'
                self._value_ = float(value)

            elif model == 'm' or model == 'g':
                if model == 'm': self._model_ = 'Monkhorst-pack' 
                elif model == 'g': self._model_ = 'Gamma'
                
                if isinstance(value, str): 
                    value = value.split()
                value = np.array(value, dtype=float)

                if value.size == 3:
                    self._value_ = np.array([value, np.zeros(3)])
                elif value.size == 6:
                    self._value_ = value.reshape(2,3)
                else:
                    raise ValueError("Gamma KPOINTS size must be 3 or 6, but input %s" %value)

            elif model == 'l':
                self._model_ = 'Line Model'
                coords = []
                symbols = []
                numbers = []
                if isinstance(value[0], str):
                    value = [k.split() for k in value]

                for line in value:
                    coords.append(np.array(line[:3], dtype=float))
                    if len(line) > 3:
                        symbols.append(line[3])
                    if len(line) > 4:
                        numbers.append(int(line[4]))

                if len(symbols) == 0:
                    symbols = ['K'] * len(coords)
                else:
                    assert len(symbols) == len(coords), "symbols != coords "
                if len(numbers) == 0:
                    numbers = number
                else:
                    assert len(numbers) == len(coords), "numbers != coords "

                self._value_ = BandPath(coords, symbols, numbers)

            elif model == 'r':
                self._model_ = 'Reciprocal'
                if isinstance(value[0], str):
                    value = [k.split() for k in value]
                value = np.array(value, dtype=float)

                if value.shape == (len(value),3):
                    self._value_ = np.c_[value,np.ones(len(value))]                
                elif value.shape == (len(value),4):
                    self._value_ = value
                else: 
                    raise ValueError("Invalid KPOINTS for Reciprocal model.")

            # Finally %
            else:
                raise ValueError("Invalid KPOINTS Model : %s" %model)
        
        else:
            raise ValueError("Invalid KPOINTS args.")

    @property
    def model(self):
        return self._model_

    @property
    def comment(self):
        return self._comment if len(self._comment_) else self._model_

    @property
    def value(self):
        return self._value_

    def get_gamma_kpoints(self, **kwargs):

        if self.model == 'kspacing':
            if 'cell' not in kwargs:
                raise ValueError('mesh kpoints to reciprocal kpoints need cell for symmetry.')
            cell = kwargs['cell']
            model = 'Gamma'
            axis = '111'
            if 'model' in kwargs:
                if isinstance(kwargs['model'], str):
                    model = kwargs['model']
                elif isinstance(kwargs['model'], tuple):
                    model, axis = kwargs['model']

            kspacing = self.value
            rec_lattice = np.linalg.inv(cell)
            mesh = np.ceil(np.linalg.norm(rec_lattice, axis=0) * 2*np.pi / kspacing).astype(int)
            for i,j in enumerate(axis):
                if j == 0:
                    mesh[i] = 1

            return Kpoints(model, mesh)

        else:
            return self

    def get_reciprocal_kpoints(self, **kwargs):
        import spglib

        if self.model == 'Reciprocal':
            return self
        
        elif self.model in ['kspacing','Gamma','Monkhorst-pack']:

            if 'cell' not in kwargs:
                raise ValueError('mesh kpoints to reciprocal kpoints need cell for symmetry.')
            cell = kwargs['cell']
            isym = kwargs['isym'] if 'isym' in kwargs else 1

            if self.model == 'kspacing':
                kspacing = self.value
                rec_lattice = np.linalg.inv(cell[0])
                mesh = np.ceil(np.linalg.norm(rec_lattice, axis=0) * 2*np.pi / kspacing).astype(int)
                shift = np.zeros(3)

            else:
                mesh = np.array(self.value[0], dtype=int)
                shift = np.array(self.value[1])
        
            mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=shift, symprec=1e-3)
        
            if isym == 1:
                index, count = np.unique(mapping, return_counts=True)
                kpts = (grid[index] + shift*(0.5,0.5,0.5)) / mesh
                results = np.c_[kpts, count]

            elif isym == 0:
                kpts = []
                weights = []
                skip = list(range(len(grid)))
                for i, coord in enumerate(grid):
                    weight = 1
                    # update index %
                    if i not in skip: continue
                    kpts.append(coord)
                    skip.remove(i)
                    # search related coord %
                    for j in skip:
                        if np.abs(coord + grid[j]).max() < 1e-8:
                            skip.remove(j)
                            weight += 1
                    weights.append(weight)
                kpts = (np.array(kpts) + shift*(0.5,0.5,0.5)) / mesh
                results = np.c_[kpts, weights]

            elif isym == -1:
                weights = np.ones(len(grid))
                kpts = (grid + shift*(0.5,0.5,0.5)) / mesh
                results = np.c_[kpts, weights]
                        
            return Kpoints("Reciprocal", results)            

        elif self.model == 'Line Model': 

            kpts = []
            for i in range(len(self.value.numbers)):
                if self.value.masks[i]:
                    kp1 = self.value.sites[i].position
                    kp2 = self.value.sites[i+1].position
                    num = self.value.numbers[i]
                    step = np.subtract(kp2,kp1)
                    kpath = [kp1 + step / num * nk for nk in range(num)]
                    kpts.extend(kpath)                    
                else:
                    kpts.append(self.value.sites[i].position)

            return Kpoints("Reciprocal", kpts)


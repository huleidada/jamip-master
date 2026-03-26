import os
import numpy as np

def load_shape(f):
    offset = None
    while True:
        line = f.readline()
        if not line: break
        if len(line.split()) == 3:
            shape = np.array(line.split(),dtype=int)
            offset = f.tell()
            break
    if offset is None:
        raise OSError("Find charge shape failed. Please check file.")

    return shape, offset 

class Chgcar(object):

    def __init__(self, path:str, offset:int, poscar:dict, shape:list):
        self.path = path
        self.offset = offset
        self.poscar = poscar
        self.shape = shape
        self._chgcar = None
        # for spin 
        self.offset_spin = None  
        self._chgcar_spin = None

    def load_charge(self, path, offset):
        with open(path,'r') as f:
            f.seek(offset)
            data = []
            num = 0
            for line in f:  
                if line.startswith('augmentation') or line.strip() == "":
                    break
                elif len(line.split()) > 0:
                    data.extend(line.split())
            data = np.array(data,dtype=float)
            assert data.size == np.prod(self.shape), f"{data.size} != {self.shape}"
        return data.reshape(self.shape[::-1]).transpose(2,1,0)

    @property
    def chgcar(self):
        """
        cache data if use this property 
        """
        if not isinstance(self._chgcar,np.ndarray):
            self._chgcar = self.load_charge(self.path, self.offset)
        return self._chgcar

    @property
    def chgcar_spin(self):
        """
        cache data if use this property 
        """
        if self.offset_spin is None:
            with open(path,'r') as f:
                f.seek(self,offset)
                _,self.offset_spin = load_shape(f) 

        if not isinstance(self._chgcar_spin,np.ndarray):
            self._chgcar_spin = self.load_charge(self.path, self.offset_spin)
        return self._chgcar_spin

    @property
    def structure(self):
        from jamip.structure import Structure

        obj = Structure()
        structure = self.poscar
        obj.comment_line = structure['comment']
        obj.lattice = structure['lattice']
        obj.atomic_coord_format = structure['type']
        obj.species_of_elements = structure['elements']
        obj.number_of_atoms = structure['numbers']
        obj.atomic_positions = structure['positions']
        return obj

    def get_fast_path_charge(self, sites): 
        '''
        sites: direct coords
        '''
        from scipy.interpolate import griddata
 
        chg = self.chgcar
        # site position in charge grid
        grid = np.mgrid[-1:2, -1:2, -1:2].reshape(3,-1).T
        path = sites * chg.shape

        # get neighbor positions of sites in charge grid
        gpath = np.around(path,0).astype(int)[:,None,:] + grid[None,:,:]
        gpath = np.unique(gpath.reshape(-1,3), axis=0)

        # move neighbor positions into base cell and interpolate
        fgpath = gpath - np.floor(gpath/chg.shape).astype(int) * chg.shape
        chggrid = chg[fgpath[:,0], fgpath[:,1], fgpath[:,2]]
        charge = griddata(gpath, chggrid, path, method='linear', rescale=True)
 
        return charge

    def get_path_charge(self, sites, method='linear'):

        from scipy.interpolate import interpn
        chg = self.chgcar
        path = (sites - np.floor(sites)) * chg.shape
        points = tuple(np.arange(i+1) for i in chg.shape)
        pbcpoints = (np.append(np.arange(i),0) for i in chg.shape)
        chg = chg[tuple(np.meshgrid(*pbcpoints, indexing='ij'))]
        charge = interpn(points, chg, path, method=method)
        return charge

    @classmethod
    def from_file(cls, path:str):

        if not os.path.exists(path):
            raise OSError("CHGFILE %s not exists!" %path)

        with open(path, 'r') as f:
            # comment
            comment=''
            string=f.readline()
            if string != "":
                comment = string.strip()
                
            scale=float(f.readline())
            
            # lattice 
            lattice=[]
            for i in range(0,3):
                lattice.append(f.readline().split())
            lattice=np.array(lattice, dtype=float) * scale
            
            # element VASP5.x
            elements=[]
            tmp=np.array(f.readline().split())
            for i in range(0,tmp.shape[0]):
                if not(tmp[i].isalpha()):
                    print('elements contain non-alphabet!')
                    exit()
            elements=tmp
            
            # numbers
            numbers=[]
            try:
                tmp=np.array([int(s0) for s0 in f.readline().split()])
                if elements.shape[0] != tmp.shape[0]:
                    print("length of numbers don't match with that of elements")
                    exit()
                numbers=tmp
            except ValueError:
                print("can't transfer literal to int type!")
                exit()
                
            # ftype
            tmp=f.readline()
            if tmp.lower().startswith('s'): # Selective dynamics
                tmp=f.readline()

            ftype=None
            if tmp.lower().startswith('c'):
                ftype='Cartesian'
            elif tmp.lower().startswith('d'):
                ftype='Direct'
            else:
                print('type of POSCAR is invalid')
                exit()
            
            # position
            natoms=sum(numbers)
            positions=[]
            for i in range(0, natoms):
                try:
                    string=f.readline().split()
                    positions.append(np.array(string[:3],dtype='float'))
                except ValueError:
                    ("can't transfer literal to float type!")
                    exit()
            positions=np.array(positions)
            if ftype == 'Cartesian':
                positions = positions*scale

            # chgcar shape
            shape, offset = load_shape(f) 

        poscar={'comment':comment,
                'lattice':lattice,
                'elements':elements,
                'numbers':numbers,
                'type':ftype,
                'positions':positions}

        return cls(path, offset, poscar, shape)

    @classmethod
    def load_all_parchg(cls, path:str):
        from pathlib import Path
        root = Path(path)
        if not root.is_dir():
            raise OSError("Need directionary")
        data = {}
        for path in root.glob('PARCHG.[0-9][0-9][0-9][0-9].[0-9][0-9][0-9][0-9]'):
            print(path)
            pc = cls.from_file(path)
            data[path.name] = pc
        return data

    @classmethod
    def write(cls, data, info:dict, output='CHGCAR'):
        from jamip.structure import Structure
        import pathlib

        path = pathlib.Path(output)
        if path.is_dir():
            path = path / 'CHGCAR'

        if isinstance(info, Structure):
            info = info.to_row()

        with open(path, 'w') as f:
            # comment line %
            f.write(info["comment"]+'\n')
            # scale line %
            f.write('  1.0\n')
            # lattice lines %
            for l in info["lattice"]:
                f.write(''.join('{0:>12.6f}'.format(c) for c in l))
                f.write('\n')
            # species line %
            f.write(' '.join('{0:>5s}'.format(e) for e in info['elements']))
            f.write('\n')
            # number of elements line %
            f.write(' '.join('{0:>5d}'.format(n) for n in info['numbers']))
            f.write('\n')
            # direct or casterain line % 
            f.write(info['type']+'\n')
            for p in info['positions']:
                f.write(''.join('{0:>10.6f}'.format(j) for j in p))
                f.write('\n')
            # blank line %
            f.write('\n')
            # data shape line %
            f.write(''.join('{0:>5d}'.format(n) for n in data.shape))
            f.write('\n')
            # chgcar lines %
            tmp = 0 
            data = data.transpose(2,1,0).reshape(-1)
            while tmp < data.size:
                f.write(''.join('{0:>19.11E}'.format(n) for n in data[tmp:tmp+5]))
                f.write('\n')
                tmp += 5


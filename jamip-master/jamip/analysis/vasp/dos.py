import os
import numpy as np
from jamip.analysis.base import Finder
from jamip.utils.utils import lazy_property
from scipy.sparse import data
from .outcar import GrepOutcar
from .doscar import Doscar
from .xml import Xml
from collections import namedtuple
from typing import Union
import pathlib

Kpath = namedtuple('Kpath', ['kpts','insert','dirs'])
Projection = namedtuple('projection', ['atoms', 'orbits','label'])

class Proj:

    def __init__(self, elements:list, allorbits:tuple):
        self.elements = elements
        self.allorbits = allorbits

    def to_proj(self, atoms:tuple, orbits:tuple, label:str=''):
        # atoms2list
        result = []
        for i in atoms:
            if isinstance(i, str):
                rows = np.where(self.elements==i)[0].tolist()
                if len(rows) == 0: 
                    raise ValueError("No matching element %s" %i)
                result.extend(rows)
            else:
                result.append(i)

        # orbits2list
        index = []
        for i,orbit in enumerate(self.allorbits):
            if orbit in orbits:
                index.append(i)
        if len(self.allorbits) == 9:
            if 'p' in orbits: index += [1,2,3]
            if 'd' in orbits: index += [4,5,6,7,8]
        if len(self.allorbits) == 16:
            if 'p' in orbits: index += [1,2,3]
            if 'd' in orbits: index += [4,5,6,7,8]
            if 'f' in orbits: index += [9,10,11,12,13,14,15]

        return Projection(atoms=list(set(result)), orbits=tuple(set(index)), label=label)


class DosFinder(Finder):

    class Result:

        def __init__(self, energy:np.ndarray, tdos:np.ndarray, pdos:Union[np.ndarray,None], 
                     fermi:float, **kwargs):
            """
            tdos: [nspin, nedos, n]
            pdos: [natom, nspin, nedos, n]
            """
            self.energy = energy
            self._tdos = tdos
            self.pdos = pdos
            self.fermi = fermi
            if 'volume' in kwargs: self.volume = kwargs['volume']
            if 'nelect' in kwargs: self.nelect = kwargs['nelect']
            if 'elements' in kwargs: self.elements = kwargs['elements']
            if 'spin' in kwargs and kwargs['spin'] is not None:
                if kwargs['spin'] == 1 and self.spin == 2:
                    self._tdos = np.sum(self._tdos,axis=0)[None,:,:]
                    self.pdos = np.sum(self.pdos,axis=1)[:,None,:,:]

        @property
        def allorbits(self):
            if self.pdos.shape[3] == 3:
                return ('s','p','d')
            elif self.pdos.shape[3] == 4:
                return ('s','p','d','f')
            elif self.pdos.shape[3] == 9:
                return ('s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'x2-y2')
            elif self.pdos.shape[3] == 16:
                return ('s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'x2-y2',
                        'fy3x2', 'fxyz', 'fyz2', 'fz3', 'fxz2', 'fzx2', 'fx3')
            else:
                raise ValueError("dostype must be ldos or mdos for projection") 
                
        @property
        def spin(self):
            return self._tdos.shape[0]

        @property
        def tdos(self):
            return self._tdos[:,:,1]

        def per_volume(self, volume=None):
            if volume == None: volume = self.volume
            self._tdos = self._tdos / volume
            self.pdos = self.pdos / volume
            return self

        def get_proj(self, value='ldos'):
            allorbits = self.allorbits

            # main %
            if isinstance(value, str):
                projection = []
                for atom in np.unique(self.elements):
                    atom_indices = np.where(self.elements==atom)[0].tolist()
                    if value[0].lower() == 'e':
                        projection.append(Projection(atoms=atom_indices, orbits=range(len(allorbits)), label=atom))

                    elif value[0].lower() == 'l':
                        if len(allorbits) in (3,4):
                            projection.append(Projection(atoms=atom_indices, orbits=(0,), label=f'{atom}-s'))
                            projection.append(Projection(atoms=atom_indices, orbits=(1,), label=f'{atom}-p'))
                            projection.append(Projection(atoms=atom_indices, orbits=(2,), label=f'{atom}-d'))
                            if len(allorbits) == 4:
                                projection.append(Projection(atoms=atom_indices, orbits=(3,), label=f'{atom}-f'))
                        if len(allorbits) in (9,16):
                            projection.append(Projection(atoms=atom_indices, orbits=(0,), label=f'{atom}-s'))
                            projection.append(Projection(atoms=atom_indices, orbits=(1,2,3), label=f'{atom}-p'))
                            projection.append(Projection(atoms=atom_indices, orbits=(4,5,6,7,8), label=f'{atom}-d'))
                            if len(allorbits) == 16:
                                projection.append(Projection(atoms=atom_indices, orbits=(9,10,11,12,13,14,15), label=f'{atom}-f'))
                        else:
                            raise ValueError("dostype error")
                    elif value[0].lower() == 'm':
                        if len(allorbits) in (9,16):
                            for i,orbit in enumerate(allorbits):
                                projection.append(Projection(atoms=atom_indices, orbits=(i,), label=f'{atom}-{orbit}'))
                        else:
                            raise ValueError("dostype error")
                    else:
                        raise ValueError("value error")

            else:
                projection = []
                proj = Proj(elements=self.elements, allorbits=allorbits)
                for val in value:
                    projection.append(proj.to_proj(*val))    

            return projection

        def projection(self, proj='ldos'):
            """
            input type 1: str
                proj = 'edos' / 'ldos' / 'mdos'
                merge dos by elements / element-lorbits / element-morbits
            input type 2: lists
                proj = ['specie', [orbits,], 'label'], like:
                proj = ['Si', ['p'], 'Si-p'], ['Si', ['px','py'], 'Si-px-py'], [[1,2,3], ['d'], 'Si[1-4]-d']

            """
            # self.data [atom, ispin, nedos, orbit] -> spin-atom-proj: [nedos, nproj, ]
            if self.pdos is None:
                raise ValueError("Data class don't contain the projected part")
            datas = []
            labels = []
            proj = self.get_proj(proj)
            pdos = self.pdos
            for p in proj:
                for ispin in range(pdos.shape[1]):
                    # natom,nspin,nedos,norbit -> nedos,norbit
                    data = np.sum(pdos[p.atoms,ispin,:,:],axis=0)
                    # nedos,norbit -> nedos'
                    data = np.sum(data[:,p.orbits],axis=1)
             
                    if ispin==0 and pdos.shape[1] == 2:
                        label = f'{p.label}-up'
                    elif ispin == 1:
                        label = f'{p.label}-down'
                        data = data * -1
                    else:
                        label = p.label
                    datas.append(data)
                    labels.append(label)
            datas = np.array(datas)

            return datas, labels

    def __init__(self,stdin=None):
        self.task = 'dos'
        self.soft = 'vasp'
        self.stdin = stdin
        self.dostype = None
        self.spin = None
    
    @property
    def dosdir(self):
        return self._stdin/'electric'/'dos' if self.file == 'jamip' else self._stdin
    
    @property
    def banddir(self):
        path = self._stdin/'electric'/'band'
        if self.file == 'jamip' and path.exists():
            return path
        return self.dosdir

    def get_fermi(self, source='DOSCAR'):
        if source.lower() == 'doscar':
            return Doscar.from_file(self.dosdir).fermi_energy
        elif source.lower() == 'xml':
            return Xml(self.dosdir).fermi_energy()

    def get_data(self, source='DOSCAR', spin=None):
        import warnings
        energy,tdos=self.get_tdos(source=source)
        try:
            energy,pdos = self.get_pdos(source=source)
        except IOError:
            warnings.warn("PDOS data not exists, did you use lorbit=11 in your calculations?")
            pdos = None
        elements = self.get_elements(source=source)
        volume = self.get_volume(source=source)
        
        return self.Result(energy = energy,
                           tdos=tdos,
                           pdos=pdos,
                           fermi=self.get_fermi(),
                           elements=elements,
                           volume=volume,
                           spin=spin)

    def get_volume(self, source='OUTCAR'):
        if source.lower() == 'xml':
            return Xml(self.dosdir).volume()
        else:
            return GrepOutcar().volume(self.dosdir)
    
    def get_elements(self, source='OUTCAR'):
        if source.lower() == 'xml':
            return Xml(self.dosdir).elements()
        else:
            return GrepOutcar().elements(self.dosdir)

    def get_tdos(self, source='DOSCAR'):
        '''
        extract total dos from DOSCAR.
        input data shape: (nedos, n), n=3,5
        output data shape: (nedos, nspin) 
        return: energy, dos
        '''
        if source.lower() == 'doscar':
            dos = Doscar.from_file(self.dosdir)._get_total_dos()
            dos_energy = dos[:,0]
            if dos.shape[-1] == 3:
                # energy tot occ -> (1, nedos)
                dos = dos[None,:,:]
            elif dos.shape[-1] == 5:            
                # energy tot_up tot_down occ_up occ_down
                # -> (2, nedos)
                dos_up = dos[:,(0,1,3)]
                dos_down = dos[:,(0,2,4)]
                dos = np.stack((dos_up,dos_down),axis=0)
            return dos_energy,dos
        elif source.lower() == 'xml':
            dos = Xml(self.dosdir)._get_total_dos()
            dos_energy = dos[0,:,0]
            # drop spin 2/3/4 in soc-dos
            if dos.shape[0] == 4:
                # energy tot occ -> (1, nedos)
                dos = dos[:1,:,:]

            return dos_energy,dos

    def get_pdos(self, source='DOSCAR'):
        '''
        extract partial dos from DOSCAR.
        input data shape: (nedos, n), n=3,5
        output data shape: (nedos, nspin) 
        return: energy, dos
        '''
        if source.lower() == 'doscar':
            dos = Doscar.from_file(self.dosdir)._get_partial_dos()
            if dos.size == 0:
                raise IOError("DOS data not exists. Did you set the parameter LORBIT ?")
         
            # normal pdos %
            dos_energy = dos[0,:,0]
            if dos.shape[-1] in (4,5,10,17): # n+1
                # energy s p d -> (natom, 1, nedos, 3)
                dos = dos[:,None,:,1:]
            elif dos.shape[-1] in (7,9,19,33): # 2n+1
                # energy s_up s_down p_up p_down d_up d_down -> (natom, 2, nedos, 3)
                dos_up = dos[:,:,1::2]
                dos_down = dos[:,:,2::2]
                dos = np.stack((dos_up,dos_down),axis=1)
            elif dos.shape[-1] in (13,37,65): # 4n+1  # skip 17=4*4+1
                # energy s_total s_mx s_my s_mz p_total p_mx p_my p_mz d_total d_mx d_my d_mz
                # -> (1, natom, 3, nedos)
                dos = dos[:,None,:,1::4]
            else:
                raise ("Unsupport DOSCAR filetype!")
            return dos_energy,dos

        elif source.lower() == 'xml':
            dos = Xml(self.dosdir)._get_partial_dos()
            dos_energy = dos[0,0,:,0]
            dos_shape = dos.shape[-1]
            self.spin = 1
            if dos.shape[1] == 4:
                # spin1 spin2 spin3 spin4 -> spin1
                dos = dos[:,:1,:,1:]
            else:
                # energy tot_up tot_down occ_up occ_down -> (2, nedos)
                dos = dos[:,:,:,1:]
            return dos_energy,dos
        
    def get_vbm(self,prec=0.01):
        tdos = self._get_tdos(self.dosdir)
        nelect = self.nelect(self.dosdir)

        if tdos.shape[-1] == 5:
            tdos = tdos[:,[0,2,4]]

        for energy,occ,tot in tdos:
            if nelect-tot <= 1e-3 and occ < 1e-1:
                return energy

        # plan B
        dos_energy = tdos[:,0]
        dos_occ = tdos[:,1]
        ifermi = sum(dos_energy < self.get_fermi())
        for i in range(ifermi):
            if dos_occ[ifermi-i] > prec:
                shift = dos_energy[ifermi-i]
                return shift

if __name__ == "__main__":
    #vf = VaspFinder('band')
    pass


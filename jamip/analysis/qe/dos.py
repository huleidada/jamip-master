import os
import numpy as np
from jamip.analysis import Finder
from .qexml import Xml
import pathlib 

class DosData:
   
    def __init__(self):
        pass

    def fermi_energy(self,path):
        #path = os.path.join(path,'dos.plt.dat')
        with open(path,'r') as f:
            data = float(f.readline().split()[-2])
        return data

    def get_tdos(self,path):
        #path = os.path.join(path,'dos.plt.dat')
        with open(path,'r') as f:
            f.readline()
            tdos = []
            for line in f:
                if '*' in line: continue
                tdos.append(line.split())
        tdos = np.array(tdos,dtype=float)
        return tdos

    def get_pdos(self,path):
        pass


class DosFinder(Finder,Xml,DosData):

    class Result:

        def __init__(self, energy:np.ndarray, tdos:np.ndarray, pdos:np.ndarray, 
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

        def get_vbm(self, prec=0.01):
            nelect = self.nelect
            estep = (self.energy[-1]-self.energy[0]) / len(self.energy)
            Ecum = np.cumsum(self.tdos) * estep 
            Ecum = Ecum * self._tdos[-1,1] / Ecum[-1]

            #for i,cum in enumerate(self._tdos[:,1]):
            for i,cum in enumerate(Ecum):
                if nelect-cum < 1e-4 :
                    #print(Ecum.shape, i, Ecum[i], nelect)
                    return self.energy[i]
        
        @property
        def tdos(self):
            return self._tdos[:,0]

        @property
        def spin(self):
            return 1
    
    def __init__(self,stdin=None):
        self.__task__ = 'dos'
        self.stdin = stdin

    @property
    def stdin(self):
        if self.__builder__ == 'jamip':
            stdin = self._stdin / 'save' / 'dos.xml'
        elif self.__builder__ == 'qeout':
            stdin = self._stdin / 'data-file-schema.xml'
        elif self.__builder__ == 'qexml':
            stdin = self._stdin
        return stdin

    @property
    def tdosdat(self):
        if self.__builder__ == 'jamip':
            file1 = self._stdin / 'pdos' / 'pdos_tot'
            file2 = self._stdin / 'dos' / 'dos.dat'
            if file1.exists(): 
                return file1
            elif file2.exists(): 
                return file2

        raise OSError("datafile not exists!" )

    @property
    def pdosdir(self):
        if self.__builder__ == 'jamip':
            path = self._stdin / 'pdos'
            if not path.is_dir():
                raise OSError("datafile not exists!" )
            return path

    @stdin.setter
    def stdin(self,path):
        self._stdin = pathlib.Path(self.seek(path))

    def get_fermi(self):
        return self.fermi_energy(self.stdin)

    def get_volume(self): 
        if self.__builder__ == 'jamip':
            lattice = Xml().lattice(self.stdin)
            volume = np.linalg.det(lattice*5.29177211/10)
            return volume

    def get_nelect(self): 
        if self.__builder__ == 'jamip':
            return Xml().nelec(self.stdin)

    def get_data(self, source=''):
        import warnings

        energy,tdos=self.get_tdos(source=source)
        try:
            elements,energy,pdos = self.get_pdos(source=source)
        except IOError:
            #warnings.warn("PDOS data not exists, did you use lorbit=11 in your calculations?")
            elements = None
            pdos = None
        volume = self.get_volume()
        nelect = self.get_nelect()
        
        return self.Result(energy = energy,
                           tdos=tdos,
                           pdos=pdos,
                           fermi=self.get_fermi(),
                           elements=elements,
                           volume=volume,
                           nelect=nelect)

    def get_tdos(self, source=''):
        dos = DosData().get_tdos(self.tdosdat)
        #self._dos_type = 'tdos'
        #self.spin = 1
        dos_energy = dos[:,0]
        dos = dos[:,1:]
        return dos_energy,dos

    def get_pdos(self, source=''):
        import re
 
        #self._dos_type = 'pdos'
        #self.orbits=['s','p','d']
        #self.spin=1
        element = set()
        pdos = {}
 
        for path in self.pdosdir.iterdir():
            result = re.findall(r'pdos_atm#\d+\(([A-Z][a-z]?)\)_wfc#\d\(([spd])\)', path.name)
            if len(result):
                elm, orbit = result[0]
                element.add(elm)
                key = '{}-{}'.format(elm, orbit)
                dos = self._get_tdos(path)
                if key not in pdos:
                    pdos[key] = dos[:,1]
                else:
                    pdos[key] += dos[:,1]
                 
        dos_energy = dos[:,0]
        tmps = []
        for i in element: 
            tmp = []
            for j in self.orbits:
                key = '{}-{}'.format(i, j)
                if key in pdos:
                    tmp.append(pdos[key])
                else:
                    tmp.append(np.zeros_like(dos_energy))
            tmps.append(tmp)
 
        pdos = np.array(tmps, dtype=float)
 
        return list(element),dos_energy,pdos

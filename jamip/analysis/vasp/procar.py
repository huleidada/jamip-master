import numpy as np
import pathlib
from .outcar import GrepOutcar
import re

class Procar(object):

    filename = 'PROCAR'

    def __init__(self, path:str, nkpts:int, nbands:int, nions:int, ispin:int=1):
        self.path = path
        self.nkpts = nkpts
        self.nbands = nbands
        self.nions = nions
        self.ispin = ispin
        self._kpoints = None
        self._bands = None
        self._procar = None

    @classmethod
    def from_file(cls, path):
        p = pathlib.Path(path)
        if p.is_dir():
            p = p/cls.filename
        if not p.exists(): 
            raise IOError("File 'EIGENVAL' not exists!") 
        path = p

        with open(path,'r') as f:
            f.readline()
            line = f.readline()

        nkpts = int(re.findall(r'k-points:\s*(\d+)', line)[0])
        nbands = int(re.findall(r'bands:\s*(\d+)', line)[0])
        nions = int(re.findall(r'ions:\s*(\d+)', line)[0])
        ispin = GrepOutcar().ispin(path.parent)
        
        return cls(path, nkpts, nbands, nions, ispin)

    @property
    def kpoints(self):
        if self._kpoints is None:
            self._kpoints = self._get_kpoint()
        return self._kpoints

    @property
    def bands(self):
        if self._bands is None:
            self._bands = self._get_band()
        return self._bands

    @property
    def procar(self):
        if self._procar is None:
            self._procar = self._get_procar()
        return self._procar

    def _get_kpoint(self,weight=False):
        pattern = re.compile(r'k-point\s*\d+\s*:\s*(-?\d\.\d+)\s*(-?\d\.\d+)\s*(-?\d\.\d+)\s*weight\s*=\s*(-?\d+\.\d*)')
        kpoints = []
        with open(self.path, 'r') as f:
            for line in f:
                if line.startswith(' k-point'):
                    result = pattern.findall(line)[0]
                    kpoints.append(result)
        kpoints = np.array(kpoints[:self.nkpts],dtype=float)

        if weight:
            return kpoints
        else:
            return kpoints[:,:3]

    def _get_band(self):
        bands = []
        band = []
        last_index = 0
        with open(self.path, 'r') as f:
            for line in f:
                if line.startswith('band'):
                    value = line.split()
                    index = int(value[1]) if value[1] != '***' else len(band)+1
                    energy = float(value[4])
                    occ = float(value[7])
                    if index > last_index:
                        last_index = index
                        band.append([energy,occ])
                    elif last_index == self.nbands:
                        bands.append(band)
                        band = [[energy,occ]]
                        last_index = index
                    else:
                        raise OSError("different nkpts in OUTCAR and PROCAR!")

        # end %
        if last_index == self.nbands and len(bands)+1 == self.ispin*self.nkpts:
            bands.append(band)
            bands = np.array(bands,dtype=float)
        else:
            raise OSError("PROCAR is incomplete!")
            #kpoints.append(result)

        assert bands.shape == (self.nkpts*self.ispin,self.nbands,2)
        bands = bands.reshape(self.ispin,self.nkpts,self.nbands,2)

        return bands

    def _get_procar(self):
        procar = []
        with open(self.path, 'r') as f:
            for line in f:
                if line.startswith('ion '):
                    for i in range(self.nions+1):
                        line = f.readline()
                        procar.append(line.split()[1:])

        assert len(procar) == self.ispin*self.nkpts*self.nbands*(self.nions+1)
        procar = np.array(procar,dtype=float).reshape(self.ispin,self.nkpts,self.nbands,self.nions+1,-1)
        
        return procar
        
    def to_wien2k(self, path, unit='Ry', efermi=0, unique=False):

        if self.ispin!=1:
            print(f'write_bandstructure_boltztrap: No idea what to do for nspin={self.ispin}')
            return False

        # eV -> Ry : 1Ry=13.6056923 
        bands = (self.bands - efermi) / 13.6056923  

        if unique:
            ikpts = self.get_unique()
            with open(path, "w") as f:
                f.write('HTE output'+'\n') # title
                f.write(str(len(ikpts))+'\n') # no. of k-points
         
                for i in ikpts:
                    f.write(("%12.8f " * 3) %tuple(self.kpoints[i]) + "%d\n" %self.nbands)
                    for j in range(self.nbands):
                        f.write("%18.8f\n" %bands[0,i,j,0])

        else:
            with open(path, "w") as f:
                f.write('HTE output'+'\n') # title
                f.write(str(self.nkpts)+'\n') # no. of k-points
         
                for i,kp in enumerate(self.kpoints):
                    f.write("%12.8f %12.8f %12.8f %d\n" %(kp[0],kp[1],kp[2],self.nbands))
                    for j in range(self.nbands):
                        f.write("%18.8f\n" %bands[0,i,j,0])
 

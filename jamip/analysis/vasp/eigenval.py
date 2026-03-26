import numpy as np
import pathlib
import re

class Eigenval(object):

    def __init__(self, path:str, nkpts:int, nbands:int, nions:int, ispin:int=1):
        self.path = path
        self.nkpts = nkpts
        self.nbands = nbands
        self.nions = nions
        self.ispin = ispin
        self._kpoints = None
        self._bands = None

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

    @classmethod
    def from_file(cls, path):
        p = pathlib.Path(path)
        if p.is_dir():
            p = p/"EIGENVAL"
        if not p.exists(): 
            raise IOError("File 'EIGENVAL' not exists!") 
        path = str(p)

        with open(path,'r') as f:
            _,_,_,ispin = f.readline().split()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            value = f.readline().split()

        ispin = int(ispin)
        nkpts = int(value[1])
        nbands = int(value[2])
        nions = int(value[0])
        
        return cls(path, nkpts, nbands, nions, ispin)

    def _get_kpoint(self, weight=True):
        kpoints = []
        with open(self.path, 'r') as f: 
            for line in f:
                if len(line.strip()) == 0:
                    kpoints.append(f.readline().split())
                    for i in range(self.nbands):
                        f.readline()

        assert len(kpoints) == self.nkpts
        kpoints = np.array(kpoints, dtype=float)
        if weight:
            return kpoints
        else:
            return kpoints[:,:3]
            
    def _get_band(self):
        bands = []
        with open(self.path, 'r') as f: 
            for line in f:
                if len(line.strip()) == 0:
                    f.readline()
                    for i in range(self.nbands):
                        bands.append(f.readline().split())

        assert len(bands) == self.nkpts * self.nbands
        if self.ispin == 1:
            bands = np.array(bands, dtype=float).reshape(1,self.nkpts,self.nbands,3)
        elif self.ispin == 2:
            bands = np.array(bands, dtype=float).reshape(self.nkpts,self.nbands,5)
            bandup = bands[:,:,(0,1,3)] 
            banddn = bands[:,:,(0,2,4)] 
            bands = np.array([bandup, banddn])

        return bands[:,:,:,1:]

    @property
    def soc_copy(self):
        ikpts = []
        for i in range(self.nkpts):
            if min(self.kpoints[i]) >= 0:
                ikpts.append(i)

        bands = self.bands[:,ikpts,:,:]
        kpoints = self.kpoints[ikpts]
        nkpts = len(ikpts)

        eigen = Eigenval(self.path, nkpts, self.nbands, self.nions, self.ispin)
        eigen._kpoints = kpoints
        eigen._bands = bands
        return eigen

    def to_wien2k(self, path, unit='Ry', efermi=0, unique=False):

        if self.ispin!=1:
            print(f'Write_bandstructure_boltztrap: No idea what to do for nspin={self.ispin}')
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
 
 

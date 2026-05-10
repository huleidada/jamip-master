import numpy as np
import pathlib
import os

class Doscar:
 
    def __init__(self, path:str, nedos:int):
        self.path = path
        self.nedos = nedos
        self._info = None
        self._total = None
        self._partial = None

    @classmethod
    def from_file(cls, path):
        p = pathlib.Path(path)
        if p.is_dir():
            p = p/"DOSCAR"
        if not p.exists(): 
            raise IOError("File 'DOSCAR' not exists!") 
        path = str(p)

        info = cls._doscar_info(path)
        nedos = int(info[2])
        return cls(path, nedos)

    @classmethod
    def _doscar_info(cls, path):
        with open(path,'r') as f:
            for i in range(5):
                f.readline()
            line = f.readline()
        return [float(i) for i in line.split()]

    @property
    def info(self):
        if self._info is None:
            self._info = self._doscar_info(self.path)
        return self._info
 
    @property
    def fermi_energy(self):
        return self.info[3]

    @property
    def emin(self):
        return self.info[1]

    @property
    def emax(self):
        return self.info[0]

    def _get_total_dos(self):
        doses = []        
        with open(self.path,'r') as f:
            for i in range(6): 
                f.readline()
            dos=[]
            for i in range(self.nedos):
                dos.append(f.readline().split())
            dos = np.array(dos,dtype=float)
        return dos

    def _get_partial_dos(self):
        doses = []        
        with open(self.path,'r') as f:
            for i in range(self.nedos+6): 
                f.readline()
            while f.readline():
                dos=[]
                for i in range(self.nedos):
                    dos.append(f.readline().split())
                doses.append(dos)
            doses = np.array(doses,dtype=float)
        return doses

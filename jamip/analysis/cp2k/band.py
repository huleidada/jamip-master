from jamip.analysis.base import Finder
from jamip.abtools.cp2k.cp2kio import CP2KIO
from collections import namedtuple
import numpy as np
import pathlib
import os
import re

Kpath = namedtuple('Kpath', ['kpts','insert','bestkpts'])
Bandside = namedtuple('BandSide', ['energy','kpoints','ikpt','iband'])
Emass = namedtuple('Emass', ['mass','energy','kpoints'])
Projection = namedtuple('projection', ['atoms', 'orbits','label'])

class BS:

    def __init__(self, params):
        self.params = params

    @property
    def lattice(self):
        cell = [None, None, None] 
        node = self.params['FORCE_EVAL']['SUBSYS']['CELL']
        for key,value in node.items():
            if key[0] == '_':
                if value[0] == 'A':
                    cell[0] = value[1:]
                elif value[0] == 'B':
                    cell[1] = value[1:]
                elif value[0] == 'C':
                    cell[2] = value[1:]
        return np.array(cell, dtype=float)

    @property
    def rec_lattice(self):
        rec_lattice = np.linalg.inv(self.lattice*5.29177211/10)*2*np.pi
        return rec_lattice

    @property
    def kpath(self):

        kpoints = []
        symbols = []
        npoints = 0
        node = self.params['FORCE_EVAL']['DFT']['PRINT']['BAND_STRUCTURE']['KPOINT_SET']
        for key,values in node.items():
            if key[0] == '_':
                if values[0] == 'SPECIAL_POINT':
                    kpoints.append(values[1:4])
                    symbols.append(re.findall('[A-Za-z0-9]',values[-1])[0])
            elif key == 'NPOINTS':
                npoints = int(values)

        inserts = [npoints] * (len(kpoints) - 1) + [1]

        return Kpath(kpts=symbols, insert=inserts, bestkpts=None)

    @classmethod
    def from_input(cls, path):
        if path.is_dir():
            path = path / 'cp2k.inp'

        params = CP2KIO.read_input(path)
        return cls(params)

    @classmethod
    def _get_info(cls, path):
        with open(path, 'r') as f:
            line = f.readline()
        result = re.findall(r'(\d+) special points, (\d+) k-points, (\d+) bands', line)
        if len(result):
            # nsp nkpt nband
            return [int(i) for i in result[0]]
        else:
            raise

    @classmethod
    def get_band(cls, path):
        if path.is_dir():
            path = path / 'cp2k.bs'

        bands = []
        with open(path, 'r') as f:
            for line in f:
                if len(line.split()) == 3:
                    bands.append(line.split())
        bands = np.array(bands, dtype=float)
        _,nkpt,nband = cls._get_info(path)
        bands = bands.reshape(nkpt, nband, 3)[...,1:]
        return bands

    @classmethod
    def get_kpoint(cls, path):
        if path.is_dir():
            path = path / 'cp2k.bs'

        kpoints = []
        with open(path, 'r') as f:
            for line in f:
                if line[0] == '#' and line.split()[1] == 'Point':
                    data = line.split()
                    ikpt = data[2]
                    ispin = data[4].rstrip(':')
                    kpt = data[5:8]
                    weight = data[8]
                    kpoints.append(kpt)
        kpoints = np.array(kpoints, dtype=float)
        return kpoints

class BandFinder(Finder):

    class Result:

        def __init__(self, bands:np.ndarray, kpoints:np.ndarray, nelect:int, **kwargs):
            self.bands = bands
            self.kpoints = kpoints
            self.metal = False
            self.nelect = nelect
            self.rec_vector = kwargs.get('rec_vector', None)
            # regroup path %
            if 'kpath' in kwargs and kwargs['kpath'] != None: 
                self.kpath = kwargs['kpath']

        def get_cbvb(self):
            bands = self.bands
            filled = int(self.nelect/2)
            for index in np.arange(filled,bands.shape[1]):
                if max(bands[:,index,1]) < 0.001:
                    return index-1,index
 
        def get_cbmvbm(self):
            bands = self.bands
            kpoints = self.kpoints
            cbvb = self.get_cbvb()
            vb, cb = cbvb
            vbm = np.argmax(bands[:,vb,0])
            cbm = np.argmin(bands[:,cb,0])

            vbm = Bandside(iband=vb, ikpt=vbm, energy=bands[vbm,vb,0], kpoints=kpoints[vbm])
            cbm = Bandside(iband=cb, ikpt=cbm, energy=bands[cbm,cb,0], kpoints=kpoints[cbm])
            # cbm %
            cvdict = {'vbm': vbm, 'cbm': cbm, 'gap': np.round(np.min(cb)-np.max(vb),6)}
            return cvdict

        def get_bandgap(self, allspin=False):
            cbvb = self.get_cbvb()
            bands = self.bands
            
            vb = bands[:,cbvb[0],0]
            cb = bands[:,cbvb[1],0]
            gap = {'indirect': np.around(np.min(cb)-np.max(vb),4),
                   'direct'  : np.around(np.min(cb-vb),4)}

            return gap

        def get_xkpt(self,continuous:bool=True):
            """
            calaulate the x coords for bandstructure plot
            kwargs:
                xticks: bool, if True, return the xticks
                continuous: bool, if True, return the xticks for continuous kpoints
                            if False, return the xticks for discrete kpoints
            return:
                xkpt: np.array, x coords for kpoints
                xticks: np.array, x coords for xticks
            """
            # nkpt -> nkpt
            delta = np.linalg.norm((self.kpoints[1:,:3]-self.kpoints[:-1,:3]) @ self.rec_vector, axis=1)
            delta = np.insert(delta,0,0)
            xkpt = np.cumsum(delta)

            # 通过计算间隔判断该点是否可导，如果不可导点的间隔为1，即该处不连续，计算此两点的间隔无意义，重置为1e-8
            if continuous:
                disc = np.where(np.abs(np.diff(delta))>1e-4)[0]
                if len(disc) > 1:
                    for i in range(1,len(disc)):
                        if disc[i]-disc[i-1] == 1:
                            k0 = disc[i-1]
                            k1 = disc[i] 
                            xkpt[k1:] -= (xkpt[k1]-xkpt[k0]+1e-8) 

            return xkpt

    def __init__(self, stdin=None):
        self.soft = 'cp2k'
        self.task = 'band'
        self.stdin = stdin

    @property
    def banddir(self):
        if self.file == 'jamip':
            stdin = self._stdin / 'nscf' / 'cp2k.bs'
        else: # 'cp2k':
            stdin = self._stdin

        return stdin

    def get_bands(self, banddir=None, source='bs'):
        """
        get band-structure data from OUTCAR

        Returns:
            ndarray: with shape (spin,kpt,band,2[energy, occupation] )
        """
        if banddir == None:
            banddir = self.banddir

        if source.lower() == "bs":
            bands = BS.get_band(banddir)
            # unit hartree > eV %
            #bands[:,:,0] *=  27.211629
            return bands
        else:
            raise ValueError("unknown source type!")
            data = func.from_file(dir) 
            return data.bands

    def get_kpoints(self, banddir=None, source='bs'):
        """
        get K-points in reciprocal lattice from OUTCAR

        Returns:
            ndarray: with shape (nkpt,3)
        """
        if banddir == None:
            banddir = self.banddir

        if source.lower() == "bs":
            return BS.get_kpoint(banddir)
            
        else:
            raise ValueError("unknown source type!")
            data = func.from_file(dir) 
            return data.kpoints

    def get_data(self, banddir=None, source='bs'):
        """
        get Result set with base func
        """
        from .output import LogFinder

        if banddir == None:
            banddir = self.banddir

        bs = BS.from_input(banddir)
        nelect = LogFinder().num_electrons(banddir)
        fermi = LogFinder().fermi_energy(banddir)
        bands = self.get_bands(banddir, source=source)
        kpoints = self.get_kpoints(banddir, source=source)
        
        if bands.shape[0] != kpoints.shape[0]:
            print("bands shape:", bands.shape)
            print("kpoints shape:", kpoints.shape)
            raise ValueError("Data shapes do not match!")

        return self.Result(bands=bands, 
                           kpoints=kpoints, 
                           nelect=nelect,
                           fermi=fermi,
                           rec_vector=bs.rec_lattice,
                           kpath=bs.kpath)


if __name__ == "__main__":
    xmlfile = "/home/kzhou/qest/TEST/Si.vasp/qerun/band.xml"

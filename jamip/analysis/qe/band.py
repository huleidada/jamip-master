from jamip.analysis import Finder
from .qexml import Xml
from collections import namedtuple
import numpy as np
import pathlib

Kpath = namedtuple('Kpath', ['kpts','insert','bestkpts'])
Bandside = namedtuple('BandSide', ['energy','kpoints','ikpt','iband'])
Emass = namedtuple('Emass', ['mass','energy','kpoints'])
Projection = namedtuple('projection', ['atoms', 'orbits','label'])

class BandData:

    def get_kpath(self,infile):
        inserts = []
        kpaths = []
        kpath = []
        with open(infile,'r') as f:
            for line in f:
                if 'K_POINTS' in line and line.split()[1] == 'crystal_b':
                    num = int(f.readline())
                    for i in range(num):
                        line = f.readline().split()
                        kpath.append(line[-1])
                        if int(line[3]) > 1:
                            inserts.append(int(line[3]))
                        else:
                            kpaths.append(kpath)
                            kpath = []

        if len(kpath): kpaths.append(kpath)

        return Kpath(kpts=kpaths, insert=inserts, bestkpts=None)

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
        #self.task = 'band'
        self.soft = 'qe'
        self.stdin = stdin

    @property
    def stdin(self):
        if self.__builder__ == 'jamip':
            stdin = self._stdin / 'save' / 'band.xml'
        elif self.__builder__ == 'qeout':
            stdin = self._stdin / 'data-file-schema.xml'
        elif self.__builder__ == 'qexml':
            stdin = self._stdin
        return stdin

    @property
    def banddir(self):
        return self.stdin

    @stdin.setter
    def stdin(self,path):
        self._stdin = pathlib.Path(self.seek(path))

    def get_kpath(self):

        if self.__builder__ == 'jamip':
            stdin = self._stdin / 'band.in'
            return BandData().get_kpath(stdin)

        else:
            raise RuntimeError('Come soon')

    def get_rec_lattice(self):
        if self.__builder__ == 'jamip':
            lattice = Xml().lattice(self.stdin)
            rec_lattice = np.linalg.inv(lattice*5.29177211/10)*2*np.pi
            return rec_lattice

    def get_bands(self, banddir=None, source='xml'):
        """
        get band-structure data from OUTCAR

        Returns:
            ndarray: with shape (spin,kpt,band,2[energy, occupation] )
        """
        if banddir == None:
            banddir = self.banddir

        if source.lower() == "xml":
            print(banddir)
            bands = Xml().get_band(banddir)
            # unit hartree > eV %
            bands[:,:,0] *=  27.211629
            return bands
        else:
            raise ValueError("unknown source type!")
            # data = func.from_file(dir) 
            # return data.bands

    def get_kpoints(self, banddir=None, source='xml'):
        """
        get K-points in reciprocal lattice from OUTCAR

        Returns:
            ndarray: with shape (nkpt,3)
        """
        if banddir == None:
            banddir = self.banddir

        if source.lower() == "xml":
            return Xml().get_kpoint(banddir)
            
        else:
            raise ValueError("unknown source type!")
            # data = func.from_file(dir) 
            # return data.kpoints

    def get_data(self, banddir=None, source='xml'):
        """
        get Result set with base func
        """
        if banddir == None:
            banddirs = self.banddir

        kpath = self.get_kpath()
        nelect = Xml().nelec(self.stdin)
        #fermi = self.get_fermi(banddir, source=source)
        bands = self.get_bands(banddir, source=source)
        kpoints = self.get_kpoints(banddir, source=source)
        if bands.shape[0] != kpoints.shape[0]:
            print("bands shape:", bands.shape)
            print("kpoints shape:", kpoints.shape)
            raise ValueError("Data shapes do not match!")

        return self.Result(bands=bands, 
                           kpoints=kpoints, 
                           nelect=nelect,
                           #fermi=fermi,
                           rec_vector=self.get_rec_lattice(),
                           kpath=kpath)


if __name__ == "__main__":
    xmlfile = "/home/kzhou/qest/TEST/Si.vasp/qerun/band.xml"
    # banddat = GrepBand()._get_band(xmlfile)
    # kpoints = GrepBand()._get_kpoint(xmlfile)

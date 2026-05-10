from jamip.analysis.base import Finder
from collections import namedtuple
import numpy as np
import pathlib
import re

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

        def __init__(self, occs:list, virts:list, **kwargs):
            self.occs = occs
            self.virts = virts
            self.unit = 'eV'

        @property
        def homo(self):
            value = np.max([occ[-1] for occ in self.occs])
            if self.unit == 'eV':
                return value * 27.21
            elif self.unit == "Hartree":
                return value

        @property
        def lumo(self):
            value = np.min([virt[0] for virt in self.virts]) 
            if self.unit == 'eV':
                return value * 27.21
            elif self.unit == "Hartree":
                return value
 
        @property
        def bandgap(self, allspin=False):
            return self.lumo - self.homo

    def __init__(self, stdin=None):
        self.task = 'band'
        self.soft = 'gaussian'
        self.stdin = stdin

    @property
    def banddir(self):
        return self.stdin / "opt.out"

#    @property
#    def stdin(self):
#        if self.__builder__ == 'jamip':
#            stdin = self._stdin / 'opt.out'
#        elif self.__builder__ == 'log':
#            stdin = self._stdin
#        return stdin

#    @stdin.setter
#    def stdin(self,path):
#        self._stdin = pathlib.Path(self.seek(path))

    def get_eigenvalues(self, banddir=None, source='log'):
        """
        get eigenvalues data from xxx.log

        Returns:
            ndarray: with shape ([[occa, virta],[occb, virtb]])
        """
        if banddir == None:
            banddir = self.banddir

        with open(banddir) as f:
            for line in f:
                if 'The electronic state is' in line:
                    mspin = line.split()[-1][0]
                    occa = []
                    virta = []
                    occb = []
                    virtb = []
                    for line in f:
                        if 'Alpha' in line:
                            if 'occ.' in line:
                                occa.extend(line.split()[4:])
                            elif 'virt.' in line:
                                virta.extend(line.split()[4:])
                        elif 'Beta' in line:
                            if 'occ.' in line:
                                occb.extend(line.split()[4:])
                            elif 'virt.' in line:
                                virtb.extend(line.split()[4:])
                        else:
                            break
   
        # # unit hartree > eV %
        #    bands[:,:,0] *=  27.211629
        if mspin == '1':
            occs = [np.array(occa, dtype=float)]
            virts = [np.array(virta, dtype=float)]
            return occs, virts 
        elif mspin == '2':
            occs = [np.array(occa, dtype=float), np.array(occb, dtype=float)]
            virts = [np.array(virta, dtype=float), np.array(virtb, dtype=float)]
            return occs, virts 
        else:
            raise ValueError('unknown mspin')


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

    def get_data(self, banddir=None, source='xml'):
        """
        get Result set with base func
        """
        if banddir == None:
            banddir = self.banddir

        occs, virts = self.get_eigenvalues(banddir)

        return self.Result(occs=occs, virts=virts) 

if __name__ == "__main__":
    xmlfile = "/home/kzhou/qest/TEST/Si.vasp/qerun/band.xml"
    #banddat = GrepBand()._get_band(xmlfile)
    #kpoints = GrepBand()._get_kpoint(xmlfile)

"""
Created on Mon Apr 15 20:57:39 2019

@author: lits
"""
from .outcar import GrepOutcar
import numpy as np
import os
import sys
from jamip.structure import read

class GrepXrays(GrepOutcar):

    def __init__(self):
        pass

    def get_formula(self,path):
        struct = read(path)
        return struct.get_formula()

    def get_crystal(self,path):
        from xrayutilities.materials.material import Crystal
        return Crystal.fromCIF(path)

    def create_xrd(self,path):
        import xrayutilities as xru
        crystal = self.get_crystal(path)
        pd = xru.simpack.PowderDiffraction(crystal)

        Am_max = max([float(pd.data[i]['r']) for i in pd.data])
        HKL = []
        for key,value in pd.data.items():
            if value['active'] is True:
                tmp = {}
                tmp['theta*2'] = value['ang']*2
                tmp['Amplitude'] = value['r'] / Am_max * 100
                tmp['d_spacing'] = pd.wavelength / (2 * np.sin(value['ang'] * np.pi / 180))
                tmp['hkl'] = key
                HKL.append(tmp)

        return HKL

    def plot_xrd(self,path,fname='xrd.png', label=None):
        import xrayutilities as xru
        from xrayutilities.simpack.powdermodel import PowderModel
        import matplotlib.pyplot as plt
        from xrayutilities.mpl_helper import SqrtAllowNegScale
        scale = SqrtAllowNegScale
        
        crystal = self.get_crystal(path)
        pd = xru.simpack.PowderDiffraction(crystal,enable_simulation=True)
        twotheta = np.arange(20,90,0.1)
#        twotheta = np.arange(27,32,0.02)
        # sim = pd.Convolve(twotheta,mode='local')
        sim = pd.Calculate(twotheta,mode='local')

        #plt.figure(figsize=(6,4))
        ax = plt.gca()

        mask = np.ones_like(twotheta, dtype=bool)
        if label is None:
            label='simulation'
        formatsim='-r'
        if isinstance(sim, PowderModel):
            simdata = sim.simulate(twotheta[mask])
            sim.plot(twotheta[mask], label=label, formatspec=formatsim, ax=ax)
        else:
            simdata = sim
            ax.plot(twotheta[mask], simdata, formatsim, label=label)

        plt.xlim(20,90)
        plt.ylim(0)
        #plt.axvline(28.33, linestyle='--')
        #plt.legend(frameon=False)

        ax.set_xlabel('2Theta (deg)')
        ax.set_ylabel('Intensity')
        ax.set_yscale('sqrt')
        plt.tight_layout()
        if fname != None:
            plt.savefig(fname)
        plt.close()
        pd.close()
        del pd
        del sim
        del crystal
        del plt

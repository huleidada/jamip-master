from jamip.analysis.vasp import OpticsFinder, BandFinder
from pymatgen.analysis.solar.slme import optics, slme
from jamip.structure import read
import numpy as np
import pytest
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



class Test_slme:

    def test_01(self,request):

        # calc slme only base on optics calculation
        calcdir = request.config.rootdir / 'testfile/calculation/slme'
        bandgap = BandFinder(calcdir).get_data(source='outcar').get_bandgap()
        of = OpticsFinder(calcdir)
        ofr = of.get_data()
        #slme1 = ofr.sl3me(5e-3, bandgap, 1.6226)
        slme1 = of.get_slme(5e-3, bandgap, shift=0, method='sl3me')
        slme2 = of.get_slme(5e-3, bandgap, shift=0, method='old')
        print('jamip-sl3me',slme1)
        print('jamip-old',slme2)
        absorb1 = np.mean(ofr.absorb(), axis=1)
        alpha_am = ofr._alpha_am()

        data = optics(calcdir / "vasprun.xml")
        energy,absorb,dirgap,indirgap = data
        slme3 = slme(*data,thickness=5e-3) 
        print('pymatgen',slme3)

        assert abs(bandgap['direct'] - dirgap) < 1e-4
        assert abs(bandgap['indirect'] - indirgap) < 1e-4
        assert sum(abs(ofr.energy - energy)) < 1e-4
        assert sum(abs(absorb1 - absorb)) / sum(absorb) < 1e-2
        assert abs(slme1 - slme3) < 1, "slme_jamip = %f\tslme_pymatgen = %f" %(slme1,slme3)

 
    def test_02(self,request):

        # calc slme with band/hseband
        calcdir = request.config.rootdir / 'testfile/calculation/slme'
        bandgap = BandFinder(calcdir).get_data(source='outcar').get_bandgap()
        # bandgap = {'direct': 1.6, 'indirect': 1.55}
        of = OpticsFinder(calcdir)
        ofr = of.get_data()
        # assert (hsegap - pbegap) == 1
        slme1 = of.get_slme(5e-3, bandgap, shift=1, method='sl3me')
        print('jamip-sl3me',slme1)
        absorb1 = np.mean(ofr.absorb(), axis=1)
        alpha_am = ofr._alpha_am()

        data = optics(calcdir / "vasprun.xml")
        energy,absorb,dirgap,indirgap = data
        dirgap += 1
        indirgap += 1
        energy += 1
        slme2 = slme(energy,absorb,dirgap,indirgap,thickness=5e-3) 
        print('pymatgen',slme2)

        assert abs(slme1 - slme2) < 1, "slme_jamip = %f\tslme_pymatgen = %f" %(slme1,slme2)

    def _03(self,request):

        # batch calc slme & plot
        calcdir = request.config.rootdir / 'testfile/calculation/slme'
        bandgap = BandFinder(calcdir).get_data().get_bandgap()
        of = OpticsFinder(calcdir)
        ofr = of.get_data()

        x = np.arange(0, 5e-3, 5e-5)
        slme1 = of.get_slme(x, bandgap, shift=0, method='sl3me')
        alpha_am = ofr._alpha_am()
        absorb1 = np.mean(ofr.absorb(),axis=1) 

        data = optics(calcdir / "vasprun.xml")
        energy,absorb,dirgap,indirgap = data
        slme2 = [slme(*data,thickness=i) for i in x]

        plt.figure(figsize=(4,3), dpi=300)
        plt.plot(x, slme1, label='jamip')
        plt.plot(x, slme2, label='pymatgen')
        plt.xlabel('thickness (cm)')
        plt.ylabel('$SLME$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('slme.png')
         
        plt.figure(figsize=(4,3), dpi=300)
        plt.axes(yscale='log')
        plt.plot(energy, absorb1, label='jamip')
        plt.plot(energy, absorb, label='pymatgen')
        plt.plot(alpha_am[:,0], alpha_am[:,2], label='am')
        plt.xlabel('energy (eV)')
        plt.ylabel('$Absorb$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('absorb.png')

if __name__ == '__main__':
    pytest.main(['-vs'])

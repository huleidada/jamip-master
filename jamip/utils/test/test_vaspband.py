from jamip.analysis.vasp import BandFinder, ProFinder, DosFinder, GrepOutcar
import numpy as np
import pytest
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pathlib


class Test_slme:

    def test_01(self,request):

        # calc band base on band calculation
        calcdir = request.config.rootdir / 'testfile/calculation/band'
        calcdir2 = request.config.rootdir / 'testfile/calculation/band_split'
 
        # total method
        b1 = BandFinder(calcdir)
        b1.get_fermi()
        b1.get_kpath()
        bd = b1.get_data()

        b2 = BandFinder(calcdir2)
        b2.get_fermi()
        b2.get_kpath()
        bd = b2.get_data()

        # get data from wavecar
        bw = BandFinder(calcdir2).get_data(source='wavecar')
        bw = BandFinder(calcdir2/'Gamma-X').get_data_from_wavecar()

        bd.get_cbvb()
        bd.get_cbmvbm()
        bd.get_metal()
        bd.get_metal_cbvb()
        bd.get_metal_cbmvbm()
        bd.get_bandgap()
        bd.get_emass()

        # check result of ikpt, energy
        b1 = BandFinder(calcdir).get_data(source='eigenval')
        b2 = BandFinder(calcdir).get_data(source='outcar')
        b3 = BandFinder(calcdir).get_data(source='procar')
        b4 = BandFinder(calcdir).get_data(source='xml')
        b5 = BandFinder(calcdir).get_data(source='wavecar')

        print(b1.bands.shape, b2.bands.shape, b3.bands.shape, b4.bands.shape, b5.bands.shape)
        assert b1.bands.shape == b2.bands.shape == b3.bands.shape  == b4.bands.shape == b5.bands.shape 
        assert b1.kpoints.shape == b2.kpoints.shape == b3.kpoints.shape == b4.kpoints.shape == b5.kpoints.shape 

        ikpts = []
        ibands = []
        energy = []
        for b in [b1,b2,b3,b4,b5]:
            cbvb = b.get_cbmvbm()
            assert np.sum(abs(cbvb['vbm'].kpoints)) < 1e-4  # Γ
            assert cbvb['vbm'].iband == 15
            assert abs(cbvb['vbm'].energy-4.7841) < 1e-4
            assert cbvb['cbm'].ikpt == 57
            assert cbvb['cbm'].iband == 16
            assert abs(cbvb['cbm'].energy-5.6387) < 1e-4
            assert abs(cbvb['gap']-0.8546) < 1e-4

        # check result of ikpt, energy
        b1 = ProFinder(calcdir).get_data(source='eigenval')
        b2 = ProFinder(calcdir).get_data(source='outcar')
        b3 = ProFinder(calcdir).get_data(source='procar')
        b4 = ProFinder(calcdir).get_data(source='xml')

        assert b1.bands.shape == b2.bands.shape == b3.bands.shape == b4.bands.shape 
        assert b1.kpoints.shape == b2.kpoints.shape == b3.kpoints.shape == b4.kpoints.shape 
        assert b1.procar.shape == b2.procar.shape == b3.procar.shape == b4.procar.shape 
 
    def test_02(self,request):

        # calc slme with band/hseband
        calcdir = request.config.rootdir / 'testfile/calculation/dos'

        # check result of ikpt, energy
        b1 = DosFinder(calcdir).get_data(source='doscar')
        b2 = DosFinder(calcdir).get_data(source='xml')
        #b2 = DosFinder(calcdir).get_data(source='outcar')

        assert b1.energy.shape == b2.energy.shape
        assert b1.tdos.shape == b2.tdos.shape
        assert b1.pdos.shape == b2.pdos.shape
        assert abs(b1.volume - b2.volume) < 1e-2

    def test_03(self,request):

        # calc slme with band/hseband
        calcdir = request.config.rootdir / 'testfile/calculation/spin_soc_f/spin'

        # check result of ikpt, energy
        b1 = DosFinder(calcdir).get_data(source='doscar')
        b2 = DosFinder(calcdir).get_data(source='xml')
        #b2 = DosFinder(calcdir).get_data(source='outcar')

        assert b1.energy.shape == b2.energy.shape
        assert b1.tdos.shape == b2.tdos.shape
        assert b1.pdos.shape == b2.pdos.shape

        b1 = BandFinder(calcdir).get_data(source='eigenval')
        b2 = BandFinder(calcdir).get_data(source='outcar')
        b3 = BandFinder(calcdir).get_data(source='procar')
        b4 = BandFinder(calcdir).get_data(source='xml')

        assert b1.bands.shape == b2.bands.shape == b3.bands.shape == b4.bands.shape 
        assert b1.kpoints.shape == b2.kpoints.shape == b3.kpoints.shape == b4.kpoints.shape 

        p1 = ProFinder(calcdir).get_data(source='xml')
        p2 = ProFinder(calcdir).get_data(source='procar')
        assert p1.procar.shape == p2.procar.shape

 
    def test_04(self,request):

        # calc slme with band/hseband
        calcdir = request.config.rootdir / 'testfile/calculation/spin_soc_f/soc'

        # check result of ikpt, energy
        b1 = DosFinder(calcdir).get_data(source='doscar')
        b2 = DosFinder(calcdir).get_data(source='xml')
        #b2 = DosFinder(calcdir).get_data(source='outcar')

        assert b1.energy.shape == b2.energy.shape
        assert b1.tdos.shape == b2.tdos.shape
        assert b1.pdos.shape == b2.pdos.shape

        b1 = BandFinder(calcdir).get_data(source='eigenval')
        b2 = BandFinder(calcdir).get_data(source='outcar')
        b3 = BandFinder(calcdir).get_data(source='procar')
        b4 = BandFinder(calcdir).get_data(source='xml')

        assert b1.bands.shape == b2.bands.shape == b3.bands.shape == b4.bands.shape 
        assert b1.kpoints.shape == b2.kpoints.shape == b3.kpoints.shape == b4.kpoints.shape 

        p1 = ProFinder(calcdir).get_data(source='xml')
        p2 = ProFinder(calcdir).get_data(source='procar')
        assert p1.procar.shape == p2.procar.shape

if __name__ == '__main__':
    #pytest.main(['-vs'])
    calcdir = '/public/home/slluo/test/jamip-test/testfile/calculation/band'
    band = BandFinder(calcdir)
    print(band.file)
    print(band.get_kpath())
    bd = band.get_data(source='xml')
    print(bd.get_cbmvbm())

    b1 = BandFinder(calcdir).get_data(source='eigenval')
    b2 = BandFinder(calcdir).get_data(source='outcar')
    b3 = BandFinder(calcdir).get_data(source='procar')
    b4 = BandFinder(calcdir).get_data(source='xml')
    for b in [b1,b2,b3,b4]:
        cbvb = b.get_cbmvbm()
        print(cbvb)
        continue
        assert cbvb['vbm'].ikpt == 0
        assert cbvb['vbm'].iband == 15
        #assert cbvb['vbm'].energy == 5.6234
        assert cbvb['cbm'].ikpt == 58
        assert cbvb['cbm'].iband == 16
        #assert cbvb['cbm'].energy == 6.3319
        assert cbvb['gap'] == 0.7085
    #.get_data()
#    print(bandgap)


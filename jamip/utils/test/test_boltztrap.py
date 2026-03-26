from jamip.analysis.vasp.boltztrap import Boltztrap
import numpy as np
import pytest
import pathlib


class Test_band_extra:

    def test_01(self,request):

        # calc band base on band calculation
        calcdir = request.config.rootdir / 'testfile/calculation/boltztrap'
 
        # total method
        b1 = Boltztrap(calcdir, mode='BTPm')
        print((calcdir/'mass.dat').exists())
        trace = b1.get_trace(calcdir)
        emass = b1.get_effective_mass(calcdir)
        print(emass)
        

    def test_02(self,request):

        # calc band base on band calculation
        calcdir = request.config.rootdir / 'testfile/calculation/boltztrap'
 
        b1 = Boltztrap(mode='BTP1')
        b1.runBTP1(calcdir)
        trace = b1.get_trace(calcdir)
        data = b1.get_data(calcdir)
        ivb, icb = data.get_cbvb_by_gap()
        emc = data.get_effective_mass(ivb, icb)
        print(emc)
        ivb, icb = data.get_cbvb_by_dos()
        emc = data.get_effective_mass(ivb, icb)
        print(emc)

        b2 = Boltztrap(mode='BTP2')
        b2.runBTP2(calcdir, overwrite=True)
        trace = b2.get_trace(calcdir)
        data = b2.get_data(calcdir)
        ivb, icb = data.get_cbvb_by_gap()
        emc = data.get_effective_mass(ivb, icb)
        print(emc)
        ivb, icb = data.get_cbvb_by_dos()
        emc = data.get_effective_mass(ivb, icb)
        print(emc)

    def xtest_03(self,request):
        from jamip.analysis.vasp.band import Emc

        # calc band base on band calculation
        calcdir = request.config.rootdir / 'testfile/calculation/emc'
 
        # total method
        bf = BandFinder(calcdir)
        emc = bf.get_emc_mass()
        print(emc)


    def xtest_03(self,request):
        from jamip.analysis.vasp.band import BandFinder

        # calc band base on band calculation
        calcdir = request.config.rootdir / 'testfile/calculation/locpot'
 
        # total method
        bf = BandFinder(calcdir)
        vacuum = bf.get_locpot_with_element()
        print(vacuum)


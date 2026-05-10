from jamip.analysis.vasp import BandFinder, ProFinder, GrepOutcar
import numpy as np
import pytest
import pathlib


class Test_band_extra:

    def xtest_01(self,request):

        # calc band base on band calculation
        calcdir = request.config.rootdir / 'testfile/calculation/band'
        calcdir2 = request.config.rootdir / 'testfile/calculation/emass'
 
        # total method
        b1 = BandFinder(calcdir)
        cbmvbm = b1.get_data().get_cbmvbm()
        emass = b1.get_emass()
        assert len(emass) == 2
        assert abs(emass['cbm'].energy-cbmvbm['cbm'].energy) < 1e-3
        assert abs(emass['vbm'].energy-cbmvbm['vbm'].energy) < 1e-3

        b2 = BandFinder(calcdir2)
        cbmvbm = b2.get_data().get_cbmvbm()
        emass = b2.get_emass()
        assert len(emass) == 6
        ecs = np.mean([emass['cbm-x'].energy,emass['cbm-y'].energy,emass['cbm-z'].energy])
        evs = np.mean([emass['vbm-x'].energy,emass['vbm-y'].energy,emass['vbm-z'].energy])
        assert abs(ecs-cbmvbm['cbm'].energy) < 1e-3
        assert abs(evs-cbmvbm['vbm'].energy) < 1e-3

        print(emass)
        emass = b2.get_emass(method='weighting')
        print(emass)
        emass = b2.get_emass(method='ploy')
        print(emass)
        

    def test_02(self,request):

        # calc band base on band calculation
        calcdir = request.config.rootdir / 'testfile/calculation/deformation'
        calcdir2 = request.config.rootdir / 'testfile/calculation/emass'
 
        # total method
        band = BandFinder(calcdir)
        dpf = band.get_deformation_potential_data()
        dp = band.get_deformation_potential()
        
        assert len(dpf) == 13
        assert len(dp) == 6
        #print(dp)
        #print(dpf)

        mu = band.get_mobility_2d(emassdir=calcdir2, dpdir=calcdir, core=('Si','1s'),axes=['x','y'])
        print(mu)
        mu = band.get_mobility_3d(emassdir=calcdir2, dpdir=calcdir, )
        #print(mu)
        mu = band.get_mobility_3d(emassdir=calcdir2, dpdir=calcdir, core=('Si','1s'))
        #print(mu)

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




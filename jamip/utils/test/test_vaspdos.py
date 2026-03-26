from jamip.analysis.vasp import BandFinder, ProFinder, GrepOutcar
import numpy as np
import pytest
import pathlib


class Test_dos_extra:

    def xtest_band_plot(self,request):
        from jamip.utils.plot import Plot
        # 一些特殊的能带数据绘制: 能带反折叠、tdm
        caldir = request.config.rootdir / 'testfile/calculation/band_unfold'
        pl = Plot()
        pl.plotter.plot_tdm(caldir, fname='tdm.png')

    def xtest_dos(self,request):
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        # calc band base on band calculation
        calcdir = request.config.rootdir / 'testfile/calculation/band_unfold'
 
        # total method
        b1 = BandFinder(calcdir)
        energy, tdos = b1.get_data().get_dos()
        assert energy.shape==(3001,)
        assert tdos.shape==(1,3001,)

        for ispin,dos in enumerate(tdos):
            plt.plot(energy, dos)
        plt.ylim(0)
        plt.savefig('band2dos.png')

    def test_jdos(self,request):
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        # calc band base on band calculation
        calcdir = request.config.rootdir / 'testfile/calculation/band_unfold'
 
        # total method
        b1 = ProFinder(calcdir)
        energy, tdos = b1.get_data().get_jdos(smear_type='gaussianinterp')
        assert energy.shape==(3001,)
        assert tdos.shape==(1,3001,)

        for ispin,dos in enumerate(tdos):
            plt.plot(energy, dos)
        plt.ylim(0)
        plt.savefig('band2jdos.png')

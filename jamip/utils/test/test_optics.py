import numpy as np
import pandas as pd
import pytest
import pathlib


class Test_optics:

    def xtest_shg(self,request):
        from jamip.analysis.vasp.shg import Shg
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        # calc band base on band calculation
        path = request.config.rootdir / 'testfile/calculation/shg'

        # get primcell transformation matrix %
        shg = Shg()
        volume = shg.get_volume_with_spacing(path=path, axis=2)
        shg.set_input(path=path, direction=[0,0,1], volume=volume, expgap=1.6921, pm=1)
        shg.run(path=path, output='shg-112.csv')

        df = pd.read_csv('shg-112.csv')
        plt.plot(df['energy'], df['real'], label='real')
        plt.plot(df['energy'], df['imag'], label='imag')
        plt.plot(df['energy'], df['abs'],  label='abs')

        plt.savefig('shg-112.png')


    def test_band_plot(self,request):
        from jamip.utils.plot import Plot
        # 一些特殊的能带数据绘制: 能带反折叠、tdm
        pl = Plot()
        caldir = request.config.rootdir / 'testfile/calculation/band'
        #pl.plotter.plot_tdm(caldir, fname='tdm.png')
        #pl.plotter.plot_cpd(caldir, fname='cpd.png')
        caldir = request.config.rootdir / 'testfile/calculation/cpl'
        pl.plotter.plot_cpd(caldir, source='wavecar',fname='cpd.png')

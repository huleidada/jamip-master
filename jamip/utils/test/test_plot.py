import pytest
from jamip.utils.plot import *
import os
import numpy as np

# 读取VASP的计算结果，绘制全部图像
# 读取QE的计算结果，绘制全部图像

class Test_plot:

    def test_01(self,tmp_path_factory,request):
        # 测试绘制能带图相关的函数
        # 绘图功能包含普通能带、投影能带、
        calcdir = request.config.rootdir / 'testfile/calculation/band'
        d = tmp_path_factory.mktemp("plot-band")
        pl = Plot()
        # pl.plotter.plot_band(calcdir, fname=d/'band.png')
        # pl.plotter.plot_fat_band(calcdir, fname=d/'proj.png')
        # pl.plotter.plot_fat_band(calcdir, ptype='base', proj='lmax', interpolation=3, max_z=2, fname=d/'proj-base.png')
        # proj = [[['Si'],['s'],'Si-s'], [['Si'],['p'],'Si-p']]
        # pl.plotter.plot_fat_band(calcdir, ptype='rb', proj=proj, fname=d/'proj-rgb.png')
        # pl.plotter.plot_single_point_dos(calcdir, proj='lmax', norm=False, fname=d/'spdos.png')
        # pl.plotter.plot_band_transitions(calcdir, search_energies=[1.5,], fname=d/'trans.png')

    def test_02(self,tmp_path_factory,request):
        # 测试绘制态密度图相关的函数
        caldir = request.config.rootdir / 'testfile/calculation/dos'
        d = tmp_path_factory.mktemp("plot-dos")
        # globalvar.dos.limit = None
        # pl = Plot()
        # pl.plotter.plot_dos(caldir, tdos=True, pdos=False, fname=d/'tdos.png')
        # pl.plotter.plot_dos(caldir, proj='lmax', tdos=False, pdos=True, fname=d/'ldos.png')
        # proj = [[['Si'],['s'],'Si-s'], [['Si'],['px'],'Si-px'], [['Si'],['py'],'Si-py'], [['Si'],['pz'],'Si-pz']]
        # pl.plotter.plot_dos(caldir, proj=proj, tdos=False, pdos=True, fname=d/'mdos.png')

    def test_03(self,tmp_path_factory,request):
        # 测试绘制吸收谱相关的函数
        caldir = request.config.rootdir / 'testfile/calculation/optics'
        d = tmp_path_factory.mktemp("plot-optics")
        pl = Plot()
        # pl.plotter.plot_absorb(caldir, ptype="absorb", fname=d/'absorb.png')        
        # globalvar.absorb.ylabel = 'refractive index'
        # pl.plotter.plot_absorb(caldir, ptype="refract", fname=d/'refract.png')
        # globalvar.absorb.ylabel = 'reflectivity'
        # pl.plotter.plot_absorb(caldir, ptype="reflect", fname=d/'reflect.png')
        # pl.plotter.plot_dielfunc(caldir, fname=d/'dielfunc.png')

    def test_04(self,tmp_path_factory,request):
        # 一些特殊的能带数据绘制: 分段能带、hse能带、gw能带、能带反折叠、tdm
        d = tmp_path_factory.mktemp("plot-band-extra")
        pl = Plot()
        # caldir = request.config.rootdir / 'testfile/calculation/band_split'
        # pl.plotter.plot_band(caldir, fname=d/'band.png') 
        # pl.plotter.plot_fat_band(caldir, ptype='base', fname=d/'proj.png') 
        # caldir = request.config.rootdir / 'testfile/calculation/hse_band'
        # globalvar.band.emin = -4
        # globalvar.band.emax = 6
        # pl.plotter.plot_hse_band(caldir, interpolation=3, fname=d/'hseband.png')
        # pl.plotter.plot_hse_band(caldir, proj='lmax', interpolation=3, fname=d/'hseproj.png')        
        caldir = request.config.rootdir / 'testfile/calculation/band_unfold'
        pl.plotter.plot_unfolding(caldir, fname=d/'unfold.png')
        pl.plotter.plot_unfolding(caldir, smear=True, fname=d/'unfold-smear.png')

    def test_05(self,tmp_path_factory,request):
        # COHP数据绘制 (COOP和ICOHP)
        caldir = request.config.rootdir / 'testfile/calculation/cohp'
        d = tmp_path_factory.mktemp("plot-cohp")
        pl = Plot()
        # globalvar.dos.limit = 0.3
        # globalvar.dos.emin = -4 
        # globalvar.dos.emax = 4
        # pl.plotter.plot_cohp(caldir, ptype='coop',fname=d/'coop.png')        
        # globalvar.dos.limit = 1
        # globalvar.dos.emin = -8 
        # globalvar.dos.emax = 8
        # # set figsize before/after plot
        # pl.plotter.set_axes(['cohp'], figsize=(6,8))
        # plt = pl.plotter.plot_cohp(caldir, ptype='pcohp',rotate=True, fname=d/'cohp.png')
        # fig = plt.gcf()    
        # fig.set_figwidth(6)
        # fig.set_figheight(8)
        # pl.save(fname=d/'cohp.png')

    def test_06(self,tmp_path_factory,request):
        # boltztrap数据绘制
        caldir = request.config.rootdir / 'testfile/calculation/boltztrap'
        d = tmp_path_factory.mktemp("plot-boltztrap")
        pl = Plot()
        # 绘制带边部分的boltztrap数据
        # pl.plotter.plot_boltztrap(caldir, ptype='Seebeck[uV/K]', cbvb='vb', fname=d/'s-vb.png')
        # pl.plotter.plot_boltztrap(caldir, ptype='Seebeck[uV/K]', cbvb='cb', fname=d/'s-cb.png')
        # pl.plotter.plot_boltztrap(caldir, ptype='powerfac[10^14]', cbvb='vb', fname=d/'pf-vb.png')
        # pl.plotter.plot_boltztrap(caldir, ptype='powerfac[10^14]', cbvb='cb', fname=d/'pf-cb.png')

    def test_07(self,tmp_path_factory,request):
        # phonopy相关数据绘制 声子谱、声子态密度、softmode、热导率、IR/Raman光谱
        caldir = request.config.rootdir / 'testfile/calculation/phonon'
        d = tmp_path_factory.mktemp("plot-phonon")
        pl = Plot()
        # pl.plotter.plot_phonon(caldir, fname=d/'phband.png')

    def test_08(self,tmp_path_factory,request):
        # md相关数据绘制 原子轨迹、能量变化、温度变化、fmsd等
        caldir = request.config.rootdir / 'testfile/calculation/md'
        d = tmp_path_factory.mktemp("plot-md")
        pl = Plot()
        # pl.plotter.plot_heat(caldir, fname=d/'md.png')
        # pl.plotter.plot_md(caldir, fname=d/'md.png')



if __name__ == '__main__':
    pytest.main(['-vs'])

import numpy as np
import pytest
import pathlib


class Test_band_extra:

    def xtest_04(self,request):
        from jamip.analysis.vasp.outcar import GrepOutcar
        from jamip.analysis.vasp.band import GrepKpath,Outcar
        from jamip.analysis.vasp.wavecar_old import Wavecar 
        from vaspwfc import save2vesta
        from jamip.structure import read
        import re

        # calc band base on band calculation
        path = request.config.rootdir / 'testfile/calculation/band_unfold'

        # get primcell transformation matrix %
        cell = read(path/'POSCAR').to_cell()
        prim = read(path/'PRIMCELL', ftype='vasp')
        primcell = prim.to_cell()
        dim = re.findall(r'-?\d+', prim.comment_line)
        assert len(dim) == 9, 'Fail to get dim from PRIMCELL'
        trans = np.array(dim, dtype=int).reshape(3,3)
        delta = cell[0] - np.dot(trans, primcell[0])
        assert np.sqrt(np.sum(delta**2)) <= 0.5, "delta %.4f out of range!" %delta

        # wavecar
        w = Wavecar.from_file(path)
#        w.trans = trans
        K,G = w.read_unfolding()
        sw = w.spectral_weight(G)
        kpath,insert = GrepKpath.read_kpath(path)
        e0, sf = w.spectral_function(sw,nedos=4000,sigma=0.01)

        # calc band base on band calculation
        path = request.config.rootdir / 'testfile/calculation/band'
        w = Wavecar.from_file(path)
        kpts = Outcar.from_file(path)._get_kpoint(weight=True)

        # chi = wfc.elf(kptw=kptw, ngrid=wfc._ngrid * 2)
        chi = w.elf(kptw=kpts[:,3], ngrid=[40, 40, 40])
        save2vesta(chi[0], lreal=True, poscar=path/'POSCAR', prefix='elf')


    def test_data_export(self,request):
        from jamip.analysis.vasp.band import GrepKpath, Outcar
        from jamip.analysis.vasp.wavecar import Wavecar
        from jamip.analysis.vasp.chgcar import Chgcar
        from vaspwfc import save2vesta, vaspwfc
        from jamip.structure import read
        import re

        # calc band base on band calculation
        path = request.config.rootdir / 'testfile/calculation/band_unfold'

        # wavecar
        w = Wavecar.from_file(path)
        primcell = w.read_primcell()
        KPTs = w.read_unfolding()
        kpath,insert = GrepKpath.read_kpath(path)

        # warning %
        cell = read(path/'POSCAR')
        delta = cell.lattice - np.dot(w.M, primcell.lattice)
        assert np.sqrt(np.sum(delta**2)) <= 0.5, "delta %.4f out of range!" %delta

        sw = w.spectral_weight(KPTs)
        e0, sf = w.spectral_function(nedos=4000,sigma=0.01)

        # calc band base on band calculation
        path = request.config.rootdir / 'testfile/calculation/band'
        cell = read(path/'POSCAR')
        w = Wavecar.from_file(path)
        w.write_elf(cell, ngrid=[40,40,40], output='ELFCAR') 
        kptw = w.KPTS[:,3]

        w = vaspwfc(path/'WAVECAR')
        chi = w.elf(kptw=kptw, ngrid=[40, 40, 40])
        save2vesta(chi[0], lreal=True, poscar=path/'POSCAR', prefix='elf2')

    def test_band_plot(self,request):
        from jamip.utils.plot import Plot
        # 一些特殊的能带数据绘制: 能带反折叠、tdm
        caldir = request.config.rootdir / 'testfile/calculation/band_unfold'
        pl = Plot()
        pl.plotter.plot_unfolding(caldir, fname='unfold.png')
        pl.plotter.plot_unfolding(caldir, smear=True, fname='unfold-smear.png')
        caldir = request.config.rootdir / 'testfile/calculation/band'
        pl.plotter.plot_tdm(caldir, fname='tdm.png')

import pytest
from jamip.utils.logger import load_yaml
from jamip.structure import read,write,Structure
from jamip.abtools.vasp.setvasp import SetVasp
from jamip.abtools.vasp.vaspflow import VaspFlow, Task
from jamip.compute.prepare import Prepare
import os
import shutil
import numpy as np

# 1. 能够访问source里配置文件
# 2. 能够正确的生成计算输入文件
# 3. 分析

@pytest.fixture()
def setvasp(tmp_path_factory,request):
    rootdir = tmp_path_factory.mktemp("vasprun")
    os.chdir(rootdir)

    func = SetVasp()
    func.tasks = "relax scf band hse_gap hse_band emass force gruneisen"
    func.structure = read(request.config.rootdir / 'testfile/structure/MoS2.vasp')
    func.potential = str(request.config.rootdir / 'testfile/potentials/vasp')
    func.program = str(request.config.rootdir / 'source/bin/bader')
    func.kpoints = 'gamma', '1 1 1'

    # copy .cluster & .incar
    shutil.copy(request.config.rootdir / 'source/env/vasp.yaml', '.incar')
    shutil.copy(request.config.rootdir / 'source/env/pbs.yaml', '.cluster')
    Prepare.links(func)
    
    return VaspFlow(func, rootdir)

class Test_vaspflow:

    structure = None

    def test_base_calc(self,request,setvasp):
        # 测试.status文件和文件生成
        flow = setvasp
        outcar = request.config.rootdir / 'testfile/calculation/relax/OUTCAR'

        # calc-1
        incar_relax = flow.tasks['relax']
        flow.set_input(incar_relax,'./relax')
        
        # create outputs (CONTCAR CHGCAR OUTCAR)
        incar_relax.structure.comment_line = 'RELAX_CONTCAR'
        write(incar_relax.structure, './relax/CONTCAR',ftype='vasp')
        shutil.copy(outcar, './relax/OUTCAR')
        with open('./relax/CHGCAR','w') as f:
            f.write('CHGCAR FILE TEST.')
        
        # create status
        stdin = './relax'
        status = incar_relax.get_status(stdin)
        flow.write_status(status, stdin)
        assert incar_relax.state == 'C'

        # calc-2
        incar_scf = flow.tasks['scf']
        incar_scf['icharg'] = 1
        incar_scf['kspacing'] = 0.189
        incar_scf.structure = flow.load_structure(stdin)
        flow.set_input(incar_scf,stdout='./scf', stdin='./relax')

        # check 
        assert incar_scf.structure.comment_line == 'RELAX_CONTCAR'
        assert os.path.exists('./scf/KPOINTS') == False
        with open('./scf/CHGCAR', 'r') as f:
            assert f.readline() == 'CHGCAR FILE TEST.'

 
    def test_base_phonon(self,tmp_path_factory,request,setvasp):
        # 测试phonopy模块与hdf5读写
        import h5py
        from jamip.analysis.vasp import PhononFinder

        # initialize
        flow = setvasp
        phonon = flow.initialize_phonon('force')
        #np.testing.assert_equal(flow.tasks['force'].dim, np.eye(3))
        
        # read hdf5
        with h5py.File("info.hdf5", "a") as h5:
            np.testing.assert_equal(h5['/structure/phonon/lattice'], flow.func.structure.lattice)
            assert h5['phonon'].attrs['symprec'] == 1e-4

        # initialize Phonopy from hdf5
        open('.status','w').close()
        ph = PhononFinder(stdin='./').get_phonon_from_info()

        
    def test_base_band(self,setvasp,request):
        from jamip.abtools.base.kpoints import Kpoints
        from jamip.abtools.vasp.vaspio import VaspIO
        from jamip.analysis.vasp import Bandside


        # 测试各种KPOINTS文件的生成
        #cifdir = request.config.rootdir / 'testfile'
        #configfile = request.config.rootdir / 'source/env/vasp.yaml'
        #params = load_yaml(configfile)
        #incar = Task(params['force'],name='force') 
        #incar.structure = read(cifdir/'MoS2.vasp')

        # initialize
        flow = setvasp
        # set kpoints by auto(seekpath) 
        flow.func.kpath.band = 15
        kpoints = flow.get_kpath('band')
        assert kpoints.value.get_insert() == 15

        # set kpoints by custom
        flow.func.kpath.band = "Line mode", (
                 r"0.0000 0.0000 0.0000 \Gamma 8",
                 r"0.5000 0.0000 0.5000 X 8",
                 r"0.5000 0.2500 0.7500 W 8",
                 r"0.0000 0.0000 0.0000 \Gamma 8",
                 r"0.5000 0.5000 0.5000 L 1")
        kpoints = flow.get_kpath('band')
        assert kpoints.value.get_insert() == 8
        VaspIO.write_kpoints(kpoints, './', name='KPOINTS')

        # set hse_band
        kpoints,kpath = flow.get_kpath('hse_band')
        VaspIO.write_kpoints(kpath, './', name='KPATH.in')
        VaspIO.write_kpoints(kpoints, './', name='KPOINTS')

        # set hse_gap
        incar = flow.tasks['hse_gap']
        incar['kspacing'] = 0.3
        #edge = self.get_band_edge(bandin)
        edge = {'vbm': Bandside(iband=5, ikpt=0, energy=0.5, kpoints=[0,0,0]),
                'cbm': Bandside(iband=6, ikpt=0, energy=2.0, kpoints=[0,0,0]),
                'gap': 1.5, }
        ibzkpt = incar.kpoints.get_reciprocal_kpoints(cell=incar.structure.to_cell(), isym=incar.get('isym',1)).value
        cbmkpt = [[*edge['cbm'].kpoints,0]]
        vbmkpt = [[*edge['vbm'].kpoints,0]]
        kpoints = np.r_[ibzkpt, cbmkpt, vbmkpt]
        incar.kpoints = Kpoints("Reciprocal",kpoints)

        # set band split 
        flow.func.kpath.band = "Line mode", (
                 r"0.0000 0.0000 0.0000 \Gamma",
                 r"0.5000 0.0000 0.5000 X",
                 r"0.5000 0.2500 0.7500 W",
                 r"0.0000 0.0000 0.0000 \Gamma",
                 r"0.5000 0.5000 0.5000 L"), 20
        kpoints = flow.get_kpath('band').value
        for k,kpts in kpoints.split().items():
            ks = Kpoints(kpts)
            VaspIO.write_kpoints(ks, name='KPOINTS')
       


if __name__ == '__main__':
    pytest.main(['-vs'])


import pytest
from jamip.utils.logger import load_yaml
from jamip.structure import read,write,Structure
from jamip.abtools.espresso.setqe import SetQE
from jamip.abtools.espresso.qeflow import QEFlow, Task
from jamip.compute.prepare import Prepare
import os
import shutil
import numpy as np

# 1. 能够访问source里配置文件
# 2. 能够正确的生成计算输入文件
# 3. 分析

@pytest.fixture()
def setqe(tmp_path_factory,request):
    rootdir = tmp_path_factory.mktemp("qerun")
    os.chdir(rootdir)

    func = SetQE()
    func.tasks = "relax scf band dos"
    func.structure = read(request.config.rootdir / 'testfile/structure/MoS2.vasp')
    func.potential = str(request.config.rootdir / 'testfile/potentials/qe')
    func.program = str(request.config.rootdir / 'source/bin/')
    func.kpoints = 'gamma', '1 1 1'

    # copy .cluster & .incar
    shutil.copy(request.config.rootdir / 'source/env/qe.yaml', '.incar')
    shutil.copy(request.config.rootdir / 'source/env/pbs.yaml', '.cluster')
    Prepare.links(func)
    
    return QEFlow(func, rootdir)

class Test_qeflow:

    structure = None

    def test_base_calc(self,request,setqe):
        # 测试.status文件和文件生成
        flow = setqe
        outcar = request.config.rootdir / 'testfile/relax/OUTCAR'

        # calc-1
        incar_relax = flow.tasks['relax']
        incar_relax.soft = 'pw.x'
        incar_relax['prefix'] = 'relax'
        incar_relax.structure.scale_factor = 1.01
        flow.set_input(incar_relax,'./relax')
        
        return 0
        # create outputs (CONTCAR CHGCAR OUTCAR)
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

 

if __name__ == '__main__':
    pytest.main(['-vs'])

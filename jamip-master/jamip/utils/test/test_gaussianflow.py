import pytest
from jamip.utils.logger import load_yaml
from jamip.structure import read,write,Structure
from jamip.abtools.gaussian.setgau import SetGaussian
from jamip.abtools.gaussian.gauflow import GaussianFlow, Task
from jamip.compute.prepare import Prepare
import os
import shutil
import numpy as np

# 1. 能够访问source里配置文件
# 2. 能够正确的生成计算输入文件
# 3. 分析

@pytest.fixture()
def setgau(tmp_path_factory,request):
    rootdir = tmp_path_factory.mktemp("gaurun")
    os.chdir(rootdir)

    func = SetGaussian()
    func.tasks = "tda=(50-50,nstates=3)"
    func.basis = "rwb97xd/6-31g(d)"
    func.structure = read(request.config.rootdir / 'testfile/structure/CH4.mol')
    func.program = 'g16'

    # copy .cluster & .incar
    shutil.copy(request.config.rootdir / 'source/env/gaussian.yaml', '.incar')
    shutil.copy(request.config.rootdir / 'source/env/pbs.yaml', '.cluster')
    Prepare.links(func)
    
    return GaussianFlow(func, rootdir)

class Test_gsflow:

    structure = None

    def test_base_calc(self,request,setgau):
        # 测试.status文件和文件生成
        flow = setgau
        #outcar = request.config.rootdir / 'testfile/OUTCAR_relax'

        # calc-1
        incar_relax = flow.tasks['opt']
        incar_relax.label = 'opt test'
        print(incar_relax.structure.connections)
        incar_relax.xc_func =  "b3lyp/6-31g(d,p)"
        incar_relax['task'] = 'opt geom=connectivity'
        flow.set_input(incar_relax,'./')
        
        return 0

        

if __name__ == '__main__':
    pytest.main(['-vs'])

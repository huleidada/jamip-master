# -*- coding: utf-8 -*-
"""
硅体系 relax → scf → band 示例任务。

用法（需已安装 jamip 且 ~/.jamip 下存在 env 模板，参见 jamip-master/install.sh）::

    cd examples/si_relax_band
    jp -r prepare -f si_relax_band.pool

生成的任务池文件可用于后续 jp -r qsub -f si_relax_band.pool（需在 .cluster 中配置真实队列）。
"""

import os
from pathlib import Path

from jamip.abtools.vasp.setvasp import SetVasp
from jamip.compute.prepare import Prepare


def jamip_input(params):
    root = Path(__file__).resolve().parent
    stub = root / "_stub"

    vasp = SetVasp()
    vasp.tasks = "relax scf band"
    vasp.xc_func = "pbe"
    vasp.kpoints = 0.2
    vasp.potential = str(stub / "paw_pbe")
    vasp.program = str(stub / "vasp_std")

    os.chdir(root)
    Prepare.cluster("pbs")

    pool_tools = Prepare.pool(vasp)
    pool_tools.set_structure(str(root / "Input"))
    Prepare.incar(vasp)
    Prepare.links(vasp)
    pool_tools.set_extra()
    pool_tools.set_potential()

    pool_arg = params.get("pool")
    if pool_arg:
        first = pool_arg[0] if isinstance(pool_arg, list) else pool_arg
        out = str(first)
    else:
        out = str(root / "si_relax_band.pool")

    pool_tools.save(file=out, mode="n")

from jamip.abtools.vasp.setvasp import SetVasp
from jamip.compute.prepare import Prepare


def jamip_input(name=None,*args, **kwargs):

    vasp=SetVasp()
    vasp.program = '/share/apps/vasp/vasp5.4.1/bin/vasp_std'    
    vasp.potential = '/share/apps/vasp/paw_pbe'

    # task %
    vasp.tasks   = 'scf band partchg emass force gruneisen optics' 
    vasp.xc_func = 'pbe' 

    vasp.force   = 1e-3         # force convergence  
    vasp.energy  = 1e-6         # energy convergence
    vasp.cutoff  = 1.3          # encut setting 
    vasp.kpoints = 0.25         # kspacing

    # Prepare %
    pool=Prepare.pool(vasp)
    pool.set_structure('Input','Output')
    pool.set_extra()
    pool.save('jptest.dat')
    Prepare.cluster('pbs')
    Prepare.incar(vasp)
    Prepare.links(vasp)

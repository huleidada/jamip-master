from jamip.abtools.espresso.setqe import SetQE
from jamip.compute.prepare import Prepare
import os

def jamip_input(name=None,*args, **kwargs):

    qe = SetQE()
    qe.program='/share/apps/qe-6.6/bin'
    qe.potential = '/share/apps/qe-6.6/PSEUDOPOTENTIALS'

    # task % 
    #qe.tasks = 'scf nscf bands relax md vc-relax vc-md dos'
    qe.tasks = 'vc-relax scf band'

    qe.force   = 1e-4         # force convergence  
    qe.energy  = 1e-5         # energy convergence
    qe.ecutwfc = 45           # wfc encut setting 
    qe.ecutrho = 360          # rho encut setting 
    qe.kpoints = 'G', '6 6 6'

    # Prepare %
    pool=Prepare.pool(qe)
    pool.set_structure('Input','Output')
    pool.save('jptest.dat')
    Prepare.cluster('pbs')
    Prepare.incar(qe)
    Prepare.links(qe)


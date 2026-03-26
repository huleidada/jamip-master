import os
import numpy as np
from dataclasses import dataclass

@dataclass
class Boltztrap_Settings:
            
    nelec= 0
    dos_type='HTSTO',
    energy_grid=0.005,
    lpfac=10
    run_type='BOLTZ'


class Boltztrap:

    settings = Boltztrap_Settings
 
    def __init__(self, builder):
        self.obj = builder

    def __getattr__(self, attr):
        return getattr(self.obj, attr)

    def diy_calculator(self):
        from os.path import join, exists, getsize
        from jamip.abtools.vasp.vaspio import VaspIO

        task_id = 'boltztrap'
        # set stdin & stdout %
        stdin = None
        if len(self.links[task_id]):
            stdin = self.tasks[self.links[task_id][0]].path
        stdout = join(self.rootdir,'electric','boltztrap')

        # add parameters %
        incar = self.tasks[task_id]
        incar.structure = self.load_structure(stdin)
        self.clear_status(task_id)

        # high-mesh scf %
        self.calculator(task_id, stdout, stdin)

        # SYMMETRY %
        if not exists(join(stdout, 'SYMMETRY')):
            VaspIO.write_symmetry(incar.structure.to_cell(), stdout)

        # run boltztrap %
        src = os.environ['HOME']+'/.jamip/bin'
        for f in ['massall.x','ani_Boltz_vasp','BoltzTrap_vasp.def']:
            if not exists(join(stdout,f)): 
                os.symlink(join(src,f), join(stdout,f))

        os.chdir(stdout)
        os.popen('./massall.x > boltztrap.dat').readline()
        os.chdir(self.rootdir)

        # check %
        if exists(join(stdout,'boltztrap.dat')) and getsize(join(stdout,'boltztrap.dat')):
            status = {'task':'boltztrap','finish':True,'success':True}
            self.write_status(status, stdout)
            incar.state = 'C'
        else:
            incar.state = 'E'

    @classmethod
    def check(self,path):
        import pathlib        
        path = pathlib.Path(path)
        if path.name == 'boltztrap':
            file = path/'mass.dat'
        else:
            file = path/'electric'/'boltztrap'/'mass.dat'
        if file.exists() and file.st_size > 0:
            print('Boltztrap calculation finish.')
            return True
        else:
            return False


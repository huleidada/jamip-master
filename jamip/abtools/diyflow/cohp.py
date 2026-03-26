import os
import numpy as np
import pathlib

class Cohp:
    '''
    default cohp parameters
    COHPstartEnergy  -20
    COHPendEnergy     10
    basisfunctions S  3s 3p
    basisfunctions Mo 4p 4d 5s
    cohpGenerator from 1.4 to 1.5 type Ga type Sr
    cohpGenerator from 2.5 to 4.0 orbitalwise
    cohpbetween atom 1 and atom 2
    '''

    def __init__(self,builder):
        self.obj = builder

    def __getattr__(self, attr):
        return getattr(self.obj, attr)

    def diy_calculator(self):
        from jamip.abtools.base.kpoints import Kpoints

        task_id = 'cohp'
        # set stdin & stdout %
        stdin = None
        if len(self.links[task_id]):
            stdin = self.tasks[self.links[task_id][0]].path
        stdout = self.rootdir / 'electric' / 'cohp'

        # add parameters %
        incar = self.tasks[task_id]
        incar.structure = self.load_structure(stdin)
        self.clear_status(task_id)
        # cohp necessary params %
        try:
            cohpfunc = incar.pop('basisfunctions')
            cohp = incar.pop('cohp')
            assert cohpfunc != None
            assert cohp != None
        except:
            raise KeyError("Missing necessary params for COHP! task exit")
        # cohp optional params %
        cohpstart = incar.pop('COHPstartEnergy', -20)
        cohpend = incar.pop('COHPendEnergy', 10)
        cohpset = incar.pop('basisSet', "pbeVaspFit2015")
        # cohp basis functions %
        #if not isinstance(cohpfunc, dict): 
        #    raise TypeError('Invalid COHP basisfunctions. Please input dict.')
        #if isinstance(cohpfunc, str):
        #    cohpfunc = {i,self.orbits[i] for i in cohpfunc.split()}
        #if isinstance(cohpfunc, list):
        #    cohpfunc = {i,self.orbits[i] for i in cohpfunc}

        # dos %
        if incar.kpoints.model not in ['Monkhorst-pack','Gamma']:
            incar.kpoints = incar.kpoints.get_gamma_kpoints(
                cell=incar.structure.lattice, model=("Gamma", "111")
                )

        if 'nbands' not in incar:
            self.get_nbands(incar)
        self.calculator(task_id, stdout, stdin)

        # write lobsterin %
        with open(stdout/"lobsterin",'w') as f:
            line = '{0:16} {1}\n'
            f.write(line.format('basisSet',cohpset))
            f.write(line.format('COHPstartEnergy',cohpstart))
            f.write(line.format('COHPendEnergy',cohpend))
            if isinstance(cohpfunc, dict):
                for elm,orbits in cohpfunc.items():
                    f.write(line.format('basisfunctions',f'{elm}  {orbits}'))
            elif isinstance(cohpfunc, list):
                for row in cohpfunc:
                    f.write(line.format('basisfunctions',row))
            f.write('\n'+ cohp)
       
        # run lobster %
        os.chdir(stdout)
        if not (stdout/'lobster').exists():
            os.symlink(os.environ['HOME']+'/.jamip/bin/lobster', stdout / 'lobster')

        os.popen("lobster > lobster.log").readline()
        os.chdir(self.rootdir)

        # check %
        if self.check(stdout):
            status = {'task':'cohp','finish':True,'success':True}
            self.write_status(status, stdout)
            incar.state = 'C'
        else:
            incar.state = 'E'

    @classmethod
    def check(self, path):
        path = pathlib.Path(path)
        if path.name == 'cohp':
            file = path / 'lobsterout'
        else:
            file = path / 'electric' / 'cohp' / 'lobsterout'
        if file.exists():
            with open(file,'r') as f:
                for line in f:
                    if line.startswith('finished'):
                        print('COHP calculation finish.')
                        return True
        return False

    @classmethod
    def get_basis_function_from_potcar(self,path):
        # TODO
        import pathlib
        pots = pathlib.Path('/public/apps/vasp/PAW_PBE.54/')
 
        with open(path/'POTCAR') as f:
            for line in f:
                if line.strip() == 'Atomic configuration':
                    f.readline()
                    for line in f:
                        results = line.split()
                        if len(line) == 0: break
                    break


import os
from ..base.check import BaseStatus
from jamip.utils.logger import load_yaml
from jamip.analysis.vasp import GrepOutcar
import pathlib

def get_max_cell_force(path):
    import numpy as np
    try:
        force = GrepOutcar().cell_force(path)
        if isinstance(force, np.ndarray) and force.size > 1:
            force = float(np.around(np.max(np.abs(force)), 3))
        else:
            force = 1000
        return force
    except:
        return 1000

class CheckStatus(BaseStatus):
    """
    cls to check the VASP
    """
    threshold = 1.00

    def __init__(self, rootdir, **kwargs):

        self.rootdir = pathlib.Path(rootdir)
        self.converge = False
        self.constrain = True  # False

    def success(self, path, task=None, **kwargs):
        """
        check vasp task run_status base on OUTCAR 
        """
        if os.path.exists(path):
            return self.get_status(path, task)
        else:
            return {'task':task,'success':False,'finish':False}

    def update_tasks(self, tasks, overwrite=[], **kwargs):
        """
        update task's path and status from .status
        """

        status = self.load_status(self.rootdir)

        for key,value in tasks.items():
            if key in status:
                value.path = status[key]['path']
                if key not in overwrite:
                    value.state = "C" if status[key]["status"] else "W"

        # batch status %
        batch_task = self.get_batch()
        if batch_task != None and batch_task in tasks:
            self.get_batch_status(tasks[batch_task])

    def is_converge(self, path, conv=None):
        go = GrepOutcar()
        if conv == None:
            conv = go.ediff(path)
        oszicar = go.oszicar(path)
        if oszicar.size == 0 or oszicar[-1,-1] <= conv:
            return True

        return False
        
    @classmethod
    def get_status(self, path:str, task=None):
        """	
        check chain: finish -> ion step -> electric step -> status
        """
        status = {'task':task, 'finish':False, 'success':False}
        
        return self.finish_check(status, path)

    @classmethod
    def finish_check(self, status, path):
        ''' step1 : finish '''

        if not (pathlib.Path(path)/"OUTCAR").exists():
            return status
        
        # case 1 : normal calculation %
        line = os.popen(f"grep 'Total CPU time used (sec)' '{path}/OUTCAR'").readline()
        if len(line) > 0:
            status['finish'] = True
            return self.ions_check(status, path)

        # case 2 : partchg or other nonscf calculation %
        line1 = os.popen(f"grep 'vasp will stop now' '{path}/OUTCAR'").readline()
        line2 = os.popen(f"grep 'VASP will stop now' '{path}/OUTCAR'").readline()
        if len(line1) > 0 or len(line2) >0 :
            status['finish'] = True
            status['success'] = True
            return status

        # case 3 : relax, to POSCAR and continue %
        line = os.popen(f"grep 'to POSCAR and continue' '{path}/vasp.log'").readline()
        if len(line) > 0:
            status['finish'] = True
            return self.ions_check(status, path)

        return status

    @classmethod
    def ions_check(self, status, path):
        ''' step2 : ions '''

        go = GrepOutcar()

        # if allow ions relax %
        if go.nsw(path) > 0:
            
            # energy converge or not %
            if go.ibrion(path) in [1,2,3] and go.isif(path) > 1:
                line = os.popen(f"grep 'reached required accuracy - stopping structural energy minimisation' '{path}/OUTCAR'").readline() 
                if len(line) == 0:
                    status['ionic'] = False
                    return status
            status['ionic'] = True

            # force converged or not % 
            if go.ediffg(path) < 0:
                status['force'] = get_max_cell_force(path)
                
        return self.electrons_check(status, path)
           
    @classmethod
    def electrons_check(self, status, path):
        ''' step3 : electrons '''

        go = GrepOutcar()

        # get last electronic step %
        line= os.popen(f"grep 'Iteration ' '{path}/OUTCAR' | tail -1").readline()
        electron_step = int(line.split('(')[-1].split(')')[0]) if line else 1000

        if electron_step < go.nelm(path):
            status['electronic'] = True
            status['success'] = True 
            return status
        elif go.ialgo(path) == 90 or go.nelmhf(path) != None:
            status['success'] = True 
            return status
        else:
            status['electronic'] = False
            return status

    def rebuild(self, tasks:dict):
        import re

        _task_ = {'electric':('dos','band','partchg','emass','stm','deformation','zpe','boltztrap','cohp',
                              'hse_gap','hse_band','meta_gap','meta_band',),
                  'magnetic':('ferro','anti','ferri'),
                  'optics':('optics','dielectric','diag','gw','bse','shg','jdos','singlet','triplet','born'),
                  'phonon':('force','softmode','gruneisen','fc2','fc3','raman'),
                  'mechanic':('elastic','poisson'),
                  'md':('tdep','shengbte')
                 }

        def get_stdout(task):
            if task == 'scf':
                return self.rootdir/task
            elif task == 'amset':
                return self.rootdir/"electric"/"deform"
            elif task == 'bader':
                return self.rootdir/"electric"/"bader"
            for i in _task_:
                if task in _task_[i]:
                    return self.rootdir/i/task


        def get_last_stdout(stdout):
            outs = []
            if stdout.exists():
                for dir in stdout.iterdir():
                    result = re.findall(r'^[A-z]+(\d+)$', dir.name)
                    if len(result) == 1 and (dir/'OUTCAR').exists(): 
                        outs.append(dir.name)
            if len(outs): return stdout/max(outs)

        if not self.rootdir.exists():
            raise IOError('Invalid calculation path')

        local_status = load_yaml(self.rootdir/'.status')
        if local_status == None: local_status={}
        # reload status %
        status_dict = {}
        for task in tasks:

            if task in ('relax','md'):
                stdout = get_last_stdout(self.rootdir/task)
                if stdout != None:
                    outdir = os.path.relpath(stdout, self.rootdir)
                    status_dict[outdir] = self.get_status(stdout, task) 

            else:
                stdout = get_stdout(task)
                #print(task, stdout, self.rootdir)
                outdir = os.path.relpath(stdout, self.rootdir)
                #print(task, stdout)

                if (stdout/'OUTCAR').exists():
                    status_dict[outdir] = self.get_status(stdout, task) 
                elif stdout.exists():
                    outs = []
                    for dir in os.listdir(stdout):
                        if (stdout/dir/'OUTCAR').exists():
                            status = self.get_status(stdout/dir, task)
                            outs.append(status['success'])
                    if len(outs) > 0 and set(outs) == {True}:
                        status_dict[outdir] = status
                elif outdir in local_status:
                    local_status.pop(outdir)

        # write_status %
        if local_status != None:
            local_status.update(status_dict)
        else:
            local_status = status_dict
        self.__save__(local_status)

        # return %
        return set([value['success'] for value in status_dict.values()]) == {True}

    @classmethod
    def load_status(cls, root):
        ''' load finish status from .status '''

        path_status = load_yaml(os.path.join(root, '.status'))
        task_status = {}
        
        if path_status != None:
            for path, value in path_status.items():
                if 'task' in value and value['task'] != None:
                    if value['task'] in task_status:
                        continue
                    if value['finish']: 
                        task = value['task']
                        task_status[task] = {'status': value['success'], 'path': path}

        return task_status

    @classmethod
    def load_relax_status(cls, root):
        ''' load converge status from .status '''

        from scipy.optimize import curve_fit

        root = pathlib.Path(root)
        path_status = load_yaml(root / '.status')
        relax_status = {}

        history = []
        if path_status != None:
            for path, value in path_status.items():
                if 'relax' in value and value['task'] == 'relax':
                    relax_status['last'] = path
                    relax_status['stat'] = value['success']
                elif path.startswith('relax') and value['finish'] == True:
                    history.append(path)

        # plan B %
        if len(relax_status) == 0 and len(history):
            path = history.pop()
            relax_status['last'] = path
            relax_status['stat'] = path_status[path]['success']
            if path == 'relax/S0':
                relax_status['stat'] = False

        if 'last' in relax_status:
            path = relax_status['last']
            relax_status['energy'] = round(GrepOutcar().free_energy(root/path), 3)

            if relax_status['stat'] == True:
                oszicar = GrepOutcar().oszicar(root/path)
                relax_status['force'] = get_max_cell_force(root/path)
                relax_status['step'] = oszicar.shape[0]
                if oszicar.size > 1:
                    # dE in last step %
                    relax_status['dE'] = '{:.2E}'.format(oszicar[-1][-1])
                if oszicar.size > 2:
                    # Energy trend %
                    Y = oszicar[-11:-1,0]
                    X = range(len(Y))
                    mod = lambda x,a,b: a*x + b
                    #try:
                    if True:
                        a,_ = curve_fit(mod,X,Y)[0]
                        if a < 0:
                            relax_status['trend'] = 'V'
                        else:
                            relax_status['trend'] = 'Λ'
                    #except:
                    #    relax_status['trend'] = '--'

        return relax_status

    def get_batch(self):

        script = pathlib.Path(self.rootdir)/'batch.sh'
        task_id = None
        if script.exists():
            with open(script,'r') as f:
                for line in f:
                    value = line.split()
                    if len(value) > 4 and value[3] == self.rootdir:
                        task_id = value[4]
        return task_id

    def get_batch_status(self, incar):

        from jamip.compute.pool import Pool
        from jamip.utils.utils import get_stdout 

        paths = []
        with Pool.open(incar.name, 'r') as pool:

            for outdir in pool.keys():
                stdout = self.rootdir / outdir
                if not stdout.exists():
                    incar.state = 'E'
                    break
                status = incar.get_status(stdout) 
                paths.append(stdout)

        if len(paths):
            stdout = get_stdout(paths)
            self.write_status(status, stdout)

        # finally, del all batch files %
        if (self.rootdir/'batch.sh').exists():
            os.remove(self.rootdir/'batch.sh')
        for file in self.rootdir.iterdir():
            if file.name.startswith('.%s' %incar.name):
                os.remove(file)

    def backup_calculations(self, task_id, stdout, stdin=None):
        import re
        import shutil

        if task_id != 'relax':
            # get index %
            indices = [] 
            if stdout.exists():
                for dir in stdout.iterdir():
                    match = re.match(r'S(-?\d+)$', dir.name)
                    if match: indices.append(int(match.group(1)))
            idx = 1
            if len(indices) and min(indices) < 1:
                idx = abs(min(indices)) + 1

            # Rename the residual calculation directory %
            newstdin = None
            for i in sorted(indices):
                if i < 0: continue
                dir = pathlib.Path(f'relax/S{i}')
                match = re.match(r'S(\d+)$',dir.name)
                if match:
                    output = dir.parent/f'S-{idx}'
                    if stdin and dir.samefile(stdin):
                        dir.rename(output)
                        newstdin = output
                    else:
                        output.mkdir()
                        for filename in ['INCAR','POSCAR','CONTCAR','OUTCAR','vasp.log']:
                            if (dir/filename).exists():
                                (dir/filename).rename(output/filename)
                        shutil.rmtree(dir)
                    idx += 1

            # write new status %
            if newstdin != None:
                status = self.get_status(newstdin, task_id)
                status['success'] = False
                self.write_status(status, newstdin)

        elif task_id == 'md':
            
            indices = [] 
            if stdout.exists():
                key = stdout.name
                for dir in stdout.parent.iterdir():
                    compiler = re.compile(rf'{key}-(\d+)')
                    if compiler.match(dir.name):
                        indices.append(int(compiler.match(dir.name).group(1)))

                # backup the last calculation %
                num = max(indices) + 1 if len(indices) else 1
                output = stdout.parent / f'{key}-{num}'
                shutil.move(stdout, output)

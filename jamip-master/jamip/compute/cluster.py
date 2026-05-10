import os
from collections import UserDict
from jamip.utils.logger import logging, full_path
from jamip.utils.logger import dump_yaml, load_yaml
import pathlib

class Cluster(UserDict):

    script = 'submit.sh'
    root = None

    def __init__(self,root=None, params=None, **kwargs):

        if root != None:
            self.__load__(root)
        if params != None:
            self.data.update(params)

    def __load__(self, root, file='.cluster'):
        '''
        load cluster configuration
        params:
            root: job submission path
            file: filename of config file
            params: update config params
        '''
        from jamip.utils.logger import load_yaml

        # load configuration file% 
        root = pathlib.Path(root).absolute()
        path = full_path(root / file)
        self.data = load_yaml(path)
        self.root = root

    def __dump__(self, root, params=None, file='.cluster'):
        from copy import deepcopy
        from jamip.utils.logger import dump_yaml

        dat = deepcopy(self.data)
        if params != None: 
            dat.update(params)
        dump_yaml(dat, os.path.join(root,'.cluster'))

    def submit(self, pool, outdir, **params):
        import socket
        import re

        self.write_script(pool, outdir)
        #if self.data['host'] == socket.gethostname():
        #    command = '{0} submit.sh | tail -1'.format(self.cmd)
        #else:
        #    script = os.path.abspath('submit.sh')
        #    command = 'ssh {0} {1} {2} | tail -1'.format(self.data['host'], self.cmd, script)
        command = '{0} submit.sh | tail -1'.format(self.cmd)
        line = os.popen(command).readline()

        # search job id %
        result = re.findall(r'\d+', line)
        if len(result):
            index = int(result[-1])
            logging.info('Submit Task : %s' %index)
            return index
        else:
            logging.error('Submit Error : %s' %line)
            exit()
   
    def write_script(self, pool, outdir, module='jamip.compute.manager'):

        from importlib import import_module
        import jinja2

        main_module = import_module(module)
        params = self.data
        params['output'] = main_module.OUTPUT
        params['script'] = main_module.__file__
        params['root'] = self.root
        params['outdir'] = outdir
        params['pool'] = full_path(pool)
        if 'env' in params and isinstance(params['env'], str):
            params['env'] = [params['env']]

        envdir = full_path("~/.jamip/env")
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(envdir), trim_blocks=True)
        temp = env.get_template(f'{self.manager}.template')
        tempout = temp.render(params)

        with open(self.script,'w') as f:
            f.writelines(tempout)

    def run(self, params, outfile=None, infile=None):
        program = params.program
        print(type(params))
        if infile != None: 
            program = f"{program} -i {infile}"

        if self.dcu != None:
            config = self.write_dcu(program)
            cmd = 'mpirun -configfile config.dcu'
        else:
            """ if don't need -np for mpirun, change following code. """
            if self.data['mpi'].split()[0] == 'mpirun':
                ntasks = params.cores if hasattr(params,'cores') else self.ntasks
                cmd = f"{self.data['mpi']} -np {ntasks} {program}"
            else:
                cmd = f"{self.data['mpi']} {program}"

        if outfile != None: cmd = cmd + ' > ' + outfile
        print(cmd)
        os.popen(cmd).readline()

    @property
    def manager(self):
        return self.data['manager'].lower()
    @property
    def mpi(self):
        return self.data['mpi']   
    @property
    def cmd(self):
        return self.data['cmd']  

    @property
    def user(self):
        return self.data.get('user', os.environ['USER'])
    @property
    def restart(self):
        return self.data.get('restart', False)
    @property
    def overwrite(self):
        return self.data.get('overwrite', False)
    @property
    def maximum(self):
        return self.data.get('maximum', 10)
    @property
    def resubmit(self):
        return self.data.get('resubmit', 'prior')
    @property
    def dcu(self):
        return self.data.get('dcu', None)
    @property
    def ntasks(self):
        if 'ntasks' in self.data:
            return self.data['ntasks'] 
        else:
            return self.nodes * self.cores
    @property
    def nodes(self):
        if 'nodes' in self.data and isinstance(self.data['nodes'], int):
            return self.data['nodes']
        else:
            return 1
    @property
    def cores(self):
        for key in ['cores', 'cpu_per_task', 'ntasks']:
            if key in self.data:
                return self.data[key]
        else:
            return 1
    @property
    def maxsleep(self):
        string = self.data.get('logtime', '10:00:00')
        coeff = [1,60,3600,86400]
        time = 0
        for i,j in zip(string.split(':')[::-1],coeff):
            time += float(i)*j
        return time
        
        


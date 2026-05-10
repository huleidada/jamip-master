import os
import pathlib

class SingleManager:

    def __init__(self, root=None):
        if root is None:
            self.rootdir = pathlib.Path.cwd().resolve()
        else:
            self.rootdir = pathlib.Path(root).resolve()

    def load_env(self):
        import subprocess
        from jamip.utils.logger import load_yaml
        
        cluster = load_yaml('.cluster')
        command = ""
        for line in cluster['env']:
            command += f'{line} >/dev/null 2>&1 && '
        command = f'bash -c "{command}env"'
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, executable="/bin/bash")
        
        for line in proc.stdout:
            key, _, value = line.decode("utf-8").partition("=")
            # skip module error 
            if key == 'BASH_FUNC_module()': continue
            os.environ[key] = value.strip()
       
        proc.communicate()  

    def submit(self, pool, outdir):
        import logging 
        from .manager import TaskManager
        from .pool import Pool

        # load func %
        func = Pool.loader(pool)[outdir]
        # go to stdout %
        stdout = self.rootdir / outdir
        if not stdout.exists():
            stdout.mkdir(parents=True)
        os.chdir(stdout)

        # update job status %
        with Pool.open(pool, 'w') as p: 
            p.update_status(outdir, 'R')
            logging.info(f"JOB start in {outdir}")

        # calculator %
        self.calculator(func,stdout)

        # update job status %
        with Pool.open(pool, 'w') as p: 
            p.update_status(outdir, 'C')
            logging.info(f"JOB end in {outdir}")

        os.chdir(self.rootdir)

    def calculator(self, func=None, stdout=None):

        from jamip.abtools.vasp.setvasp import SetVasp
        from jamip.abtools.vasp.vaspflow import VaspFlow
        from jamip.abtools.espresso.setqe import SetQE
        from jamip.abtools.espresso.qeflow import QEFlow
        from jamip.abtools.cp2k.setcp2k import SetCP2K
        from jamip.abtools.cp2k.cp2kflow import CP2KFlow
        from jamip.abtools.gaussian.setgau import SetGaussian
        from jamip.abtools.gaussian.gauflow import GaussianFlow
        from jamip.compute.pool import Pool 
        
        if isinstance(func, SetVasp):
            VaspFlow(func, self.rootdir).vasp_calculator()
        elif isinstance(func, SetQE):
            QEFlow(func, self.rootdir).qe_calculator()
        elif isinstance(func, SetCP2K):
            CP2KFlow(func, self.rootdir).cp2k_calculator()
        elif isinstance(func, SetGaussian):
            GaussianFlow(func, self.rootdir).gaussian_calculator()
        else:
            raise ValueError ("only VASP and QE WorkFlow is valid ...")	

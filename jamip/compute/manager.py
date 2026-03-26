import os
import re
import logging
from jamip.compute.cluster import Cluster
from jamip.compute.launch import LaunchTasks
from jamip.utils.logger import full_path, run_logger, cluster_info

OUTPUT = '.output'
# step 0: open the jobs pool % 
# step 1: check current jobs and analysis % 
# step 2: update status of current jobs % 
# step 3: update resource % 
# step 4: load unfinished jobs % 
# step 5: submit jobs % 
class PBSManager:
    
    @classmethod
    def get_task_by_user(self,user=None,**kwargs):
        import pandas as pd
        if user is None:
            user = os.environ['USER']
        lines = os.popen(f'qstat -u {user}').readlines()
        data = []

        if len(lines) > 0 : 

            # normally 10th slice is run-status %
            for i,line in enumerate(lines):
                if re.match('Job',line):
                    index = line.split().index('S')-1
                    break
         
            # get self tasks %
            # JOB STATUS: Q R E C H %
            for line in lines[i:]:
                if re.match(r'\d+',line):
                    status = line.split()[index]
                    if status not in ['C','E']:
                        jobid = re.match(r'\d+',line).group()
                        path = self.get_task_by_id(jobid)
                        data.append([jobid,status,path])

        return pd.DataFrame(data, columns=['id','status','path'])


    @classmethod
    def get_task_by_id(self,jobid=None,**kwargs):
        lines = os.popen(f'qstat -f {jobid}').readlines()
        path = ''
        for i,line in enumerate(lines):
            if 'PBS_O_WORKDIR=' in line:
                line = line.split('PBS_O_WORKDIR=')[-1]
                if ',' in line:
                    path = line.split(',')[0]
                if len(path) == 0:
                    path += line.rstrip()
                    for line in lines[i+1:]:
                        if line[0] != '\t': break
                        if ',' in line:
                            path += line.lstrip().split(',')[0]
                            break
                        else:
                            path += line.strip()

            if len(path) > 0:
                break

        return path

    @classmethod
    def get_host_by_id(self,jobid,**kwargs):
        lines = os.popen(f'qstat -f {jobid}').readlines()
        host = None
        for i,line in enumerate(lines):
            # 这里查到的是任务执行的目录
            if 'exec_host' in line:
                hosts = re.findall(r'exec_host\s*=\s+([A-Za-z0-9-]+)',line)
                if len(hosts) > 0:
                    return hosts[0]
            # 这里查到的是任务提交的目录
            elif 'PBS_O_HOST' in line:
                host = re.findall(r'PBS_O_HOST\s*=\s*([A-Za-z0-9-\.]+),',line)
                return host

        return host

class LSFManager:

    @classmethod
    def get_task_by_user(self,user,**kwargs):
        import pandas as pd
        if user is None:
            user = os.environ['USER']
        lines = os.popen(f'bjobs -u {user}').readlines()

        if len(lines) > 1 :

            # normally 3th slice is run-status %
            for i,line in enumerate(lines):
                if re.match('JOBID',line):
                    index = line.split().index('STAT')
                    break

            data = []
            # get self tasks %
            maps = {'RUN':'R','PEND':'Q','HOLD':'H','EXIT':'E','DONE':'C'}
            # JOB STATUS: Q(PEND) R(RUN) E C H %
            for line in lines[i:]:
                if re.match(r'\d+',line):
                    status = line.split()[index]
                    if status in ['RUN','PEND']:
                        jobid = re.match(r'\d+',line).group()
                        path = self.get_task_by_id(jobid)
                        data.append([jobid, maps[status], path]) 

        return pd.DataFrame(data, columns=['id','status','path'])

    @classmethod
    def get_task_by_id(self,jobid=None,**kwargs):
        lines = os.popen('bjobs -l %s | grep -A10 CWD' %jobid).readlines()
        path = ''
        for i,line in enumerate(lines):
            if 'CWD <' in line:
                line = line.split('CWD <')[-1]
                if '>' in line:
                    path = line.split('>')[0]
                if len(path) == 0:
                    path += line.rstrip()
                    for line in lines[i+1:]:
                        if '>' in line:
                            path += line.split('>')[0].lstrip()
                            break
                        else:
                            path += line.strip()
            if len(path) > 0:
                break

        if path.startswith('$'):
            path = os.path.expandvars(path)
        return path

class SLURMManager:

    @classmethod
    def get_task_by_user(self,user,**kwargs):        
        import pandas as pd
        if user is None:
            user = os.environ['USER']
        if kwargs.get('host'):
            lines = os.popen(f'ssh {kwargs["host"]} squeue -u {user}').readlines()
        else:
            lines = os.popen(f'squeue -u {user}').readlines()

        data = []
        if len(lines) > 1 : 
            # normally 10th slice is run-status %
            for i,line in enumerate(lines):
                if re.match('JOBID',line.lstrip()):
                    index = line.split().index('ST')
                    break

            # get self tasks %
            data = []
            maps = {'R':'R','PD':'Q','HOLD':'H','EXIT':'E','DONE':'C'}
            for line in lines[i:]:
                if re.match(r'\d+',line.lstrip()):
                    status = line.split()[index]
                    if status in ['R','PD']:
                        jobid = re.match(r'\d+',line.lstrip()).group()
                        path = self.get_task_by_id(jobid)
                        data.append([jobid, maps[status], path])

        return pd.DataFrame(data, columns=['id','status','path'])

    @classmethod
    def get_task_by_id(self,jobid=None,**kwargs):
        lines = os.popen('scontrol show job '+jobid).readlines()
        path = ''
        for i,line in enumerate(lines):
            if 'WorkDir' in line:
                path = line.split('WorkDir=')[-1].rstrip()
                '''
                if '>' in line:
                    path = line.split('>')[0]
                if len(path) == 0:
                    path += line.rstrip()
                    for line in lines[i+1:]:
                        if '>' in line:
                            path += line.split('>')[0].lstrip()
                            break
                        else:
                            path += line.strip()
                '''
            if len(path) > 0:
                break

        return path

    @classmethod
    def get_host_by_id(self,jobid,**kwargs):
        lines = os.popen(f'scontrol show job {jobid}').readlines()
        host = None
        for i,line in enumerate(lines):
            if 'BatchHost' in line:
                line = line.split('BatchHost=')[-1].rstrip()
                break

        return host

class DonauManager:

    @classmethod
    def get_task_by_user(self,user=None,**kwargs):
        import pandas as pd
        if user is None:
            user = os.environ['USER']
        lines = os.popen(f'djob -u {user}').readlines()
        data = []

        maps = {'RUNNING':'R','PENDING':'Q','HOLD':'H','EXIT':'E','DONE':'C'}
        if len(lines) > 0 : 
            # normally 10th slice is run-status %
            for i,line in enumerate(lines):
                if re.match('JOB_ID',line):
                    index = line.split().index('JOB_STATE')
                    break
         
            # get self tasks %
            # JOB STATUS: Q R E C H %
            for line in lines[i:]:
                if re.match(r'\d+',line):
                    status = maps[line.split()[index]]
                    if status in ['R','Q']:
                        jobid = re.match(r'\d+',line).group()
                        path = self.get_task_by_id(jobid)
                        data.append([jobid,status,path])

        return pd.DataFrame(data, columns=['id','status','path'])

    @classmethod
    def get_task_by_id(self, jobid=None, **kwargs):
        lines = os.popen(f'djob -L {jobid}').readlines()
        path = ''
        for i,line in enumerate(lines):
            if 'EXEC_PATH' in line:
                path = line.split()[1]
        return path

    @classmethod
    def get_host_by_id(self, jobid, **kwargs):
        lines = os.popen(f'djob -L {jobid}').readlines()
        host = None
        for i,line in enumerate(lines):
            if 'TASK_EXEC_NODES' in line:
                host = line.split()[1]

        return host


class TaskManager:

    @classmethod
    def from_yaml(cls, root):
        cluster = Cluster(root)
        obj = cls(cluster.manager)
        obj.rootdir = os.path.abspath(root)
        obj.cluster = cluster
        return obj 
    
    def __init__(self, manager=None, **kwargs):

        if manager.lower() == 'pbs':
            self.manager = PBSManager
        elif manager.lower() == 'lsf':
            self.manager = LSFManager
        elif manager.lower() == 'slurm':
            self.manager = SLURMManager
        self.manager.name = manager.lower()
        self.rootdir = None
         
    def get_task_by_user(self,user=None,**kwargs):
        if user is None and self.rootdir != None:
            user = self.cluster['user']
        return self.manager.get_task_by_user(user,**kwargs)

    def get_task_by_id(self,jobid,**kwargs):
        return self.manager.get_task_by_id(jobid,**kwargs)

    def get_host_by_id(self,jobid,**kwargs):
        return self.manager.get_host_by_id(jobid,**kwargs)

    def get_queue_num(self):

        if self.rootdir is None:
            raise ValueError("Missing info")
        if self.cluster.get('ssh', False):
            taskdict = self.get_task_by_user(self.cluster['user'], host=self.cluster['host'])
        else:
            taskdict = self.get_task_by_user(self.cluster['user'])
        total_num = len(taskdict)
        group_num = 0
        for path in taskdict['path']:
            if path.startswith(self.rootdir):
                group_num += 1 
        return total_num, group_num 

    def calculator(self, poolfile:str, stdout:str, outdir:str):

        from jamip.abtools.vasp.setvasp import SetVasp
        from jamip.abtools.vasp.vaspflow import VaspFlow
        from jamip.abtools.espresso.setqe import SetQE
        from jamip.abtools.espresso.qeflow import QEFlow
        #from jamip.abtools.cp2k.setcp2k import SetCP2K
        #from jamip.abtools.cp2k.cp2kflow import CP2KFlow
        from jamip.abtools.gaussian.setgau import SetGaussian
        from jamip.abtools.gaussian.gauflow import GaussianFlow
        from jamip.compute.pool import Pool 
        
        fpool = full_path(poolfile)
        #outdir = os.path.relpath(stdout, self.rootdir)
        func = Pool.loader(fpool)[outdir]

        if isinstance(func, SetVasp):
            VaspFlow(func, self.rootdir).vasp_calculator()
        elif isinstance(func, SetQE):
            QEFlow(func, self.rootdir).qe_calculator()
        #elif isinstance(func, SetCP2K):
        #    CP2KFlow(func, self.rootdir).cp2k_calculator()
        elif isinstance(func, SetGaussian):
            GaussianFlow(func, self.rootdir).gaussian_calculator()
        else:
            raise ValueError ("only VASP and QE WorkFlow is valid ...")	

    def unique(self, stdout:str):
        ''' job repeat check '''
        run_num = 0
        for path in self.get_task_by_user()['path']:
            if path == stdout:
                run_num += 1
        return True if run_num <= 1 else False

    def resubmit(self, poolfile:str, stdout):
        ''' submit next job '''
        # get job number %
        total_num, group_num = self.get_queue_num()
        maximum = self.cluster.maximum
        if maximum == 0:
            next_num = 0
        elif total_num == 0:
            next_num = 1
        elif self.cluster.resubmit == 'prior':
            next_num = maximum - group_num + 1
        elif self.cluster.resubmit == 'mini':
            next_num = max(maximum-total_num, 1-group_num) + 1
        # submit %
        logging.info("Next %s Task will be submited." %next_num)
        newjob = {'run':'qsub', 'pool':poolfile, 'maximum':next_num}
        print(newjob)
        os.chdir(self.rootdir)
        LaunchTasks(newjob,*[stdout])
         

def main(root:str, outdir:str, pool:str):

    # load overwrite/restart %
    run_logger()
    stdout = os.getcwd()
    if not os.path.samefile(stdout, os.path.join(root, outdir)):
        logging.error("The program did not enter the expected directory. exit...")
        exit()

    manager = TaskManager.from_yaml(root)
    if manager.unique(stdout):
        # get cluster info %
        cluster_info(manager.cluster.cores)
        # start calculation %
        manager.calculator(pool, stdout, outdir)
        # next calculation %
        manager.resubmit(pool, outdir)
    else:
        logging.warning("There is another program running in the current directory")
        # next calculation %
        manager.resubmit(pool, False)
 

# main program % 
if __name__ == '__main__':
    import sys

    pool   = sys.argv[3] 
    outdir = sys.argv[2]
    root   = sys.argv[1] 
    main(root, outdir, pool)

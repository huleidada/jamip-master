import os
from ..base.check import BaseStatus
from pathlib import Path

class CheckStatus(BaseStatus):
    """
    cls to check the QE
    """

    def __init__(self, rootdir, *args, **kwargs):

        self.rootdir = rootdir

    def success(self, stdout, task=None, **kwargs):
        """
        check guassian task run_status base on task.out
        """
        # file check %
        import shutil
        
        outfile = Path(stdout) / f"{task}.out"
        if outfile.exists():
            status = self.get_status(outfile, task)
            # xmlfile = stdout+'.xml'
            # if exists(xmlfile):
            #     shutil.copy(xmlfile,self.rundir)
            #     os.remove(xmlfile)
        else:
            status = {'task':task,'finish':False,'success':False}

        return status

    def update_tasks(self, tasks, overwrite=[], **kwargs):

        status = self.load_status(self.rootdir)

        for key,value in tasks.items():
            if key in status:
                value.path = status[key]['path']
                if key not in overwrite:
                    value.state = "C" if status[key]["status"] else "W"

    @classmethod
    def get_status(self, path, task=None):
        """     
        check chain: finish -> ion step -> electric step -> status
        """
        status = {'task':task, 'finish':False, 'success':False}
        path = Path(path) / f'{task}.out'

        return self.finish_check(status, path)

    def write_status(self, status, path):
        """
        base status update function
        """
        from os.path import relpath, join

        data = self.__load__()
        key = relpath(join(path, status['task']), self.rootdir)
        # remove original finish status %
        if status['task'] in ['relax','scf']:
            for i in list(data.keys()):
                if data[i]['task'] == status['task']:
                    data.pop(i)
        # update input status %
        if key in data:
            data.pop(key)
        data[key] = status 

        self.__save__(data)
            
    @classmethod
    def finish_check(self, status, path):
        ''' step1 : finish '''

        line = os.popen(f"grep 'Job cpu time' '{path}'").readline()
        if len(line) > 0:
            status['finish'] = True
            #status['success'] = True 
            return self.ions_check(status, path) 

        return status

    @classmethod
    def ions_check(self, status, path):
        '''step2 : relax '''
        import re
        # opt check
        lines = os.popen(f"grep -B5 'Optimization completed.' '{path}'").readlines()
        if len(lines): 
            status['success'] = True
            converged = []
            for line in lines[:4]:
                converged.append(line.split()[-1])
            if converged[0] == "YES" and  converged[1] == "YES":
                status['force'] = True
            if converged[2] == "YES" and  converged[3] == "YES":
                status['ions'] = True
            return status

        # sp check 
        lines = os.popen(f"grep -A1 'SCF Done' '{path}'").readlines()
        if len(lines):
            status['success'] = True
            conv = re.search(r'Conv=(\d.\d+)D(-?\d+)', lines[1])
            conv = f"{conv.group(1)}E{conv.group(2)}"
            status['conv'] = float(conv)
            return status

        return status

    @classmethod
    def rebuild_status(self,root,tasks):
        """
        rebuild .status base on calculation files. 
        """
        raise 

    @classmethod
    def load_status(cls, root):

        from jamip.utils.logger import load_yaml

        # load status %
        root = Path(root)
        path_status = load_yaml(root / '.status')
        task_status = {}

        if path_status != None:
            for path, value in path_status.items():
                if 'task' in value and value['task'] != None:
                    if value['finish']:
                        task = value['task']
                        task_status[task] = {'status': value['success'], 'path': path}

        return task_status

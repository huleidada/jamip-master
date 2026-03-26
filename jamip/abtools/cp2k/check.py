from ..base.check import BaseStatus
from jamip.utils.logger import load_yaml
import pathlib
import os

class CheckStatus(BaseStatus):
    """
    cls to check the CP2K
    """

    def __init__(self, rootdir, *args, **kwargs):

        self.rootdir = pathlib.Path(rootdir)

    def success(self, path, task=None, **kwargs):
        """
        check espresso task run_status base on task.out
        """

        if path.exists():
            return self.get_status(path, task)
        else:
            return {'task':task,'finish':False,'success':False}

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
        path = pathlib.Path(path) / 'cp2k.log' 
        return self.finish_check(status, path)

    def write_status(self, status, path):
        """
        base status update function
        """
        from os.path import relpath

        data = self.__load__()
        key = relpath(path, self.rootdir)
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
        line = os.popen(f"grep 'PROGRAM ENDED' '{path}'").readline()
        print(f"grep 'PROGRAM ENDED' '{path}'")
        print(line)
        if len(line) > 0:
            status['finish'] = True
            return self.ions_check(status, path)

        return status

    @classmethod
    def ions_check(self, status, path):
        '''step2 : relax '''
        line1 = os.popen(f"grep -R 'CELL   OPTIMIZATION' {path}").readline()
        line2 = os.popen(f"grep 'GEOMETRY OPTIMIZATION COMPLETED' {path}").readline()
        if len(line1) == 0 or len(line2):
            status['success'] = True
        return status

    def rebuild(self,tasks):
        """
        rebuild .status base on calculation files. 
        """
        import re

        _task_ = {'electric':('dos','band','partchg''deformation','zpe','boltztrap'),
                  'mechanic':('elastic','poisson'),
                 }

        def get_stdout(task):
            if task in ['scf','relax']:
                return self.rootdir/task
            for i in _task_:
                if task in _task_[i]:
                    return self.rootdir/i/task


        def get_last_stdout(stdout):
            outs = []
            if stdout.exists():
                for dir in stdout.iterdir():
                    result = re.findall(r'^[A-z]+(\d+)$', dir.name)
                    if len(result) == 1 and (dir/'cp2k.log').exists(): 
                        outs.append(dir.name)
            if len(outs): return stdout/max(outs)

        if not self.rootdir.exists():
            raise IOError('Invalid calculation path')

        local_status = load_yaml(self.rootdir/'.status')
        if local_status == None: local_status={}
        # reload status %
        status_dict = {}
        for task in tasks:

            stdout = get_stdout(task)
            outdir = os.path.relpath(stdout, self.rootdir)

            if (stdout/'cp2k.log').exists():
                status_dict[outdir] = self.get_status(stdout, task) 
                print(self.get_status(stdout, task))
            elif stdout.exists():
                outs = []
                for dir in os.listdir(stdout):
                    if (stdout/dir/'cp2k.log').exists():
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

        from jamip.utils.logger import load_yaml

        path_status = load_yaml(os.path.join(root, '.status'))
        task_status = {}

        if path_status != None:
            for path, value in path_status.items():
                if 'task' in value and value['task'] != None:
                    if value['finish']:
                        task = value['task']
                        task_status[task] = {'status': value['success'], 'path': path}

        return task_status

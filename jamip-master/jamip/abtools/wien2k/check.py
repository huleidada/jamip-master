from ..base.check import BaseStatus
import pathlib
import os

class CheckStatus(BaseStatus):
    """
    cls to check the QE
    """

    def __init__(self, rootdir, *args, **kwargs):

        self.rootdir = rootdir

    def success(self, stdout, task=None, **kwargs):
        """
        check espresso task run_status base on task.out
        """
        # file check %
        import shutil

        outfile = pathlib.Path(self.rundir) / f'{task}.out'
        if outfile.exists():
            status = self.get_status(outfile, task)
            stdout = pathlib.Path(stdout)
            xmlfile = stdout.parent / f'{stdout.name}.xml'
            if xmlfile.exists():
                shutil.copy(xmlfile,self.rundir)
                os.remove(xmlfile)
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
        path = pathlib.Path(path) / f'{task}.out' 

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

        line = os.popen("grep 'JOB DONE' '%s'" %path).readline()
        if len(line) > 0:
            status['finish'] = True
            status['success'] = True 

        return status

    @classmethod
    def ions_check(self, status, path):
        '''step2 : relax '''
        line = os.popen("grep 'End final coordinates' '%s'" %path).readline()
        if len(line): status['success'] = True
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

        path_status = load_yaml(os.path.join(root, '.status'))
        task_status = {}

        if path_status != None:
            for path, value in path_status.items():
                if 'task' in value and value['task'] != None:
                    if value['finish']:
                        task = value['task']
                        task_status[task] = {'status': value['success'], 'path': path}

        return task_status

        '''
        # reset vasptask - opt&scf %
        if task.relax != None and task.relax.finish == False:
            if 'relax' in status and status.relax.finish: 
                task.relax.finish = True

        if task.scf != None and task.scf.finish == False:
            if 'scf' in status and status.scf.finish:
                task.scf.finish = True

        # reset vasptask - nonscf %
        if not overwrite:
            for key in TaskBuilder().property:
                if key not in task: continue
                for prop in task[key]:
                    if task[key][prop].finish == False:
                        if prop in status and status[prop].finish:
                            task[key][prop].finish = True

        # reset stdin %
        if 'scf' in status and status.scf.finish:
            stdin = status.scf.path
        elif 'relax' in status and status.relax.finish:
            stdin = status.relax.path
        else:
            return None

        return os.path.join(root,stdin)
        '''

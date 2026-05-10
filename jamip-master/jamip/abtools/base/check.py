import os
import ruamel
from os.path import exists, join

#    yml = ruamel.yaml.YAML()
#    yml.indent(sequence=3)
#    yml.dump(data, f)
#    yml = ruamel.yaml.YAML(typ='safe', pure=True)
#    data = yml.load(f)

class BaseStatus:

    """
    class of task status
    """

    def __load__(self):
        """
        read Status from self.rootdir/.status
        """

        data = None
        if exists(join(self.rootdir, '.status')):
            try:
                with open(join(self.rootdir, '.status'), 'r') as f:
                    yml = ruamel.yaml.YAML(typ='safe', pure=True)
                    data = yml.load(f)
            except:
                pass
        if data == None:
            data = {}

        return data
 
    def __save__(self, data):
        """
        save Status to sale.rootdir/.status
        """

        with open(join(self.rootdir, '.status'), 'w+') as f:
            yml = ruamel.yaml.YAML()
            yml.indent(sequence=3)
            yml.dump(data, f)

    def write_status(self, status, path):
        """
        base status update function
        """
        data = self.__load__()
        key = os.path.relpath(path, self.rootdir)
        # remove original finish status %
        if status['task'] in ['relax','scf']:
            for i in list(data.keys()):
                if status['task'] == data[i].get('task'):
                    data.pop(i)
        # remove task label if sub-nscf-calculation %
        elif not key.endswith(status['task']):
            status.pop('task')
        # Update status dictionary at the end of the file %
        if key in data:
            data.pop(key)
        data[key] = status 

        self.__save__(data)

    def error_status(self, error, path):
        """
        write error task to status 
        """
        data = self.__load__()
        key = os.path.relpath(path, self.rootdir)
        data[key] = {'error':error,'finish':False,'success':False}
        self.__save__(data)

    def right_status(self, tasks):
        """
        update tasks to right
        """
        data = self.__load__()
        if isinstance(tasks, str):
            tasks = [tasks]

        for key,value in data.items():
            if value['task'] in tasks:
                value['finish'] = True
                value['success'] = True
        self.__save__(data)

    def clear_status(self, tasks):
        """
        clear tasks from Status
        """
        data = self.__load__()
        if isinstance(tasks, str):
            for path in list(data.keys()):
                if tasks in path.split('\\')[:2]:
                    data.pop(path)
      
        elif isinstance(tasks, list):
            for key in list(data.keys()):
                if data[key]['task'] in tasks:
                    data.pop(key)

        self.__save__(data)



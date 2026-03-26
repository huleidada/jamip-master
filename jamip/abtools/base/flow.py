import h5py
from collections import UserDict, defaultdict
import logging


class WorkFlow:

    class Links(UserDict):
        
        @property
        def nodes(self):
            rev = []
            for root,children in self.items():
                rev.append(root)  
                rev.extend(children)
            return set(rev)

        @property
        def reverse(self):
            rev = defaultdict(list)
            for root,children in self.items():
                for child in children:
                    rev[child].append(root)
            return rev

    def refresh(self, taskname):
        
        # W&H 
        if self.tasks[taskname].state in ["W", "H"]:
            tmp = []
            for parent in self.links[taskname]:
                tmp.append(self.tasks[parent].state) 
            if "W" in tmp or "H" in tmp or "E" in tmp:
                self.tasks[taskname].state = "H" 
            else:
                self.tasks[taskname].state = "W" 

        # E&C
        update = []
        if self.tasks[taskname].state in ["E", "C", "D"]:
            update = self.links.reverse[taskname]
        return update

    def launch(self):

        task_id = None
        for i,task in self.tasks.items():
            if task.state == "W":
                task_id = i
                break

        if task_id != None:
            self.run(task_id)
            queue = [task_id]
            for i in queue:
                r = self.refresh(i)
                queue.extend(r)
        else: 
            logging.info("JOB end.")

    def run(self):
        
        raise TypeError("MetaClass")

    def sinfo(self, key:str, structure):

        with h5py.File("info.hdf5", "a") as h5:
 
            if key in h5: del h5[key]
            g = h5.create_group(key)
            g['lattice'] = structure.lattice
            g['elements'] = structure.get_elements()
            g['positions'] = structure.get_positions(type='direct')

    def minfo(self, key:str, structure):

        with h5py.File("info.hdf5", "a") as h5:
 
            if key in h5: del h5[key]
            g = h5.create_group(key)
            g['elements'] = structure.get_elements()
            g['positions'] = structure.get_positions(type='cartesian')
           
            #h5[key] = structure.get_positions(type='direct')
            #h5[key].attrs['elements'] = structure.get_elements()
            #h5[key].attrs['lattice'] = structure.lattice

    def info(self, key:str, value:dict):

        with h5py.File("info.hdf5", "a") as h5:

            if key in h5: del h5[key]
            #h5.create_dataset(key, data=json.dumps(value))
            d = h5.create_dataset(key, dtype="f")
            for k,v in value.items():
                d.attrs[k] = v 

    @property
    def state(self):
        '''
        > W: waiting
        > C: completed
        > H: hold
        > E: error
        > R: running
        '''
        return {task:value.state for task, value in self.tasks.items()}

    @property
    def allstate(self):
        state = [value.state for value in self.tasks.values()]
        return True if "W" in state else False







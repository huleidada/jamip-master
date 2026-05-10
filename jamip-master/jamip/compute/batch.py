import os
import logging
from jamip.utils.logger import cluster_info, run_logger
from jamip.compute.cluster import Cluster
from jamip.compute.pool import Pool

OUTPUT = '.batch'
# step 0: open the jobs pool % 
# step 1: check current jobs and analysis % 
# step 2: update status of current jobs % 
# step 3: update resource % 
# step 4: load unfinished jobs % 
# step 5: submit jobs % 

class TaskBatch:

    def __init__(self, root:str, stdout:str, task_id:str, program:str):
        from socket import gethostname

        self.cluster = Cluster(root)
        self.program = program 
        self.task_id = task_id
        self.rootdir = stdout
        self.host = gethostname()
 
    def calculator(self):
        from jamip.compute.manager import TaskManager

        cluster_info(self.cluster.cores)
        # submit tasks %
        while self.get_all_status() is True:
            self.launch()

        # callback %
        manager = TaskManager.from_yaml(root)
        unique = manager.unique(stdout)
        status = self.get_final_status()
        logging.info('{host}: [unique={unique}, final_status={status}]'.format(host=self.host, unique=unique, status=status))
        if unique or status:
            outdir, pool = self.get_script()
            manager.calculator(pool, stdout)
            manager.resubmit(pool, outdir)


    def launch(self):
        import json
        from os.path import join, exists

        with Pool.open(self.task_id, 'w') as pool:

            for outdir in sorted(pool.items(),key=lambda v:v[1]['prior']):
                outdir = outdir[0]
                value = pool[outdir]

                if value['status'] == 'W' and value['prior'] > 0:
                    stdout = join(self.rootdir, outdir)    # absolute path %
                    if not exists(stdout):
                        raise OSError("What's up.")

                    # Mark the tasks before calculation start
                    pool.update_status(outdir, status='R') 
                    logging.info("Submit %s in %s" %(outdir, self.host))
                    self.run(stdout)
            
                    # Update dbm after calculation finish 
                    value = pool[outdir]
                    value['status'] = 'C'
                    value['prior'] -= 1
                    # host %
                    if len(self.host) > len(value['host']):
                        value['host'] = self.host[-len(value['host']):]
                    else:
                        value['host'] = self.host.ljust(len(value['host']))
                    pool[outdir] = value
                    try:
                        pool.data.sync()
                    except:
                        logging.warning('Pool sync block ! Waiting ... ')


    def run(self, stdout:str):

        os.chdir(stdout)
        cmd = "{mpi} {program} > run.log".format(mpi=self.cluster.run, program=self.program)
        os.popen(cmd).readline()
        os.chdir(self.rootdir)

    def get_all_status(self):
        with Pool.open(self.task_id, 'r') as pool:
            status = [value['status'] for key, value in pool.items()]
        return True if "W" in status else False

    def get_final_status(self):
        with Pool.open(self.task_id, 'r') as pool:
            status = [value['status'] for key, value in pool.items()]
        return set(status) == {'C'}

    def get_script(self):

        script = os.path.join(self.rootdir, 'submit.sh')
        with open(script,'r') as f:
            for line in f:
                value = line.split()
                if len(value) > 4 and value[2] == self.cluster.root:
                    outdir = value[3]
                    pool = value[4]
                    return outdir, pool

# main program % 
if __name__ == '__main__':
    import sys

    program = sys.argv[4]
    task_id = sys.argv[3] 
    stdout = sys.argv[2]
    root   = sys.argv[1] 

    # os.chdir(stdout)
    assert stdout == os.getcwd()
    run_logger()
    TaskBatch(root, stdout, task_id, program).calculator()

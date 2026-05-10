import os
import json
import logging
import pathlib

class LaunchTasks(object):

    def __init__(self, params=None, *args, **kwargs):
        
        cmd = params.pop('run')
        if cmd == 'prepare':
            self.__prepare__(params)

        elif cmd == 'qsub':
            self.__launch__(params,*args)

        elif cmd == 'single':
            self.__single__(params,*args)

        elif cmd == 'skip':
            self.__skip__(params,*args)

    def __prepare__(self, params=None, *args, **kwargs):
        import sys

        emit_json = params.pop('json', False)
        if emit_json:
            os.environ['JAMIP_PREPARE_JSON'] = '1'

        try:
            if os.path.isfile('input.py'):
                sys.path.append(os.getcwd())
                from input import jamip_input 
            else:
                raise IOError ('input.py not exists..')
        except:
            raise IOError ('please construct input.py firstly..')

        try:
            jamip_input(params)
            if emit_json:
                self.__prepare_emit_json(params)
        finally:
            os.environ.pop('JAMIP_PREPARE_JSON', None)

    def __prepare_emit_json(self, params):
        """Write prepare result as JSON to stdout (for scripting)."""
        from .pool import Pool

        pool_arg = params.get('pool')
        if pool_arg:
            first = pool_arg[0] if isinstance(pool_arg, list) else pool_arg
            pool_path = pathlib.Path(first).resolve()
        else:
            pool_path = pathlib.Path('pool/jamip').resolve()

        tasks = []
        with Pool.open(pool_path, 'r') as pool:
            for key in sorted(pool.keys()):
                row = dict(pool[key])
                tid = row.get('id')
                if isinstance(tid, str):
                    try:
                        row['id'] = int(tid.strip())
                    except ValueError:
                        pass
                tasks.append({'path': key, **row})

        doc = {
            'ok': True,
            'run': 'prepare',
            'pool_file': str(pool_path),
            'task_count': len(tasks),
            'tasks': tasks,
        }
        print(json.dumps(doc, ensure_ascii=False, indent=2))

    def get_pool(self, pool, **kwargs):
        from jamip.utils.logger import full_path

        if isinstance(pool, list):
            pool = pool[0]

        return full_path(pool)
	
    def __launch__(self, params=None, submitter=None, **kwargs):
	
        # load pool %
        from .cluster import Cluster
        from .pool import Pool
        import pathlib
        import socket

        root = pathlib.Path.cwd()
        fpool = self.get_pool(**params)
        pool = Pool(pool=fpool)
		
        # load cluster %
        cluster = Cluster(root)

        # check cluster environment % 
        if 'cluster' in params:
            cluster.update(params['cluster'])

        # check maximum number of tasks %
        maximum = cluster['maximum']
        if 'maximum' in params:
            maximum = params['maximum']

        # check restart/overwrite % 
        if 'overwrite' in params and params['overwrite']:
            cluster['overwrite'] = True
        if 'restart' in params and params['restart']:
            cluster['restart'] = True

        with pool.open(fpool, 'w') as p:
            # update task status %
            if submitter == None:
                # submit from manager, update .cluster %
                cluster['user'] = os.environ['USER']
                cluster['host'] = socket.gethostname() 
                cluster.__dump__(root)
            elif submitter:
                # submit from cluster, update pool %
                p.update_status(submitter, 'C')
 
            # qsub the jobs % 
            n = 1
            
            # according to the prior order %  
            for outdir in sorted(p.items(),key=lambda v:v[1]['prior']):
 
                if n > maximum: break
                outdir = outdir[0]          # relative path & key % 
                value = p[outdir]                    # real-time value %
                if value['status'] == 'W' and (value['prior'] > 0 or cluster.restart is True):
                    if p.update_status(outdir, 'R') != 'W': continue
                    stdout = root / outdir    # absolute path %
                    if not stdout.exists(): 
                        stdout.mkdir(parents=True)
                    # TODO: restart
                    # elif cluster.restart and exists(stdout+'/.status'):
                    #     self.__restart__(stdout)
                    # os.system('touch %s/.wait' %stdout) 
 
                    # submit the jobs % 	
                    os.chdir(stdout)
                    jobid = cluster.submit(fpool, outdir)
                    p.update_jobid(outdir, jobid)
                    #logging.info("JOBID : %s" %jobid)
                    os.chdir(root)
                    n += 1     		
           
        if n == 1:
            logging.info("The task pool is empty or all tasks have been submitted.")


    def __single__(self, params, root=None, outdir=None):
	
        from .single import SingleManager 
        from .pool import Pool

        if root is None:
            root = os.getcwd()
        fpool = self.get_pool(**params)
        pool = Pool(pool=fpool)
        single = SingleManager(root)
        single.load_env()

        with pool.open(fpool, 'r') as p:
            ordered = sorted(p.items(), key=lambda v: v[1]['prior'])

        for outdir, value in ordered:
            if value['status'] in ['W', 'R'] and value['prior'] > 0:
                single.submit(fpool, outdir)
            else:
                logging.info("JOB Finished : %s" % outdir)
	
    def __skip__(self, params, outdir=None):

        from .pool import Pool
        from .cluster import Cluster

        root = os.getcwd()
        fpool = self.get_pool(**params)
        pool = Pool(pool=fpool)
        if 'maximum' in params:
            maximum = params['maximum']
        else:
            cluster = Cluster(root)
            maximum = cluster['maximum']

        with pool.open(fpool, 'w') as p:

            n=1
            # according to the prior order %  
            for outdir in sorted(p.items(),key=lambda v:v[1]['prior']):

                if n > maximum: break
                outdir = outdir[0]          # relative path & key % 
                value = p[outdir]                    # real-time value %
                if value['status'] == 'W' and value['prior'] > 0:
                    p.update_jobid(outdir, 0)
                    n+=1
            print(f"Skip job {n-1}")


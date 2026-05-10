from jamip.compute.pool import Pool
import pathlib

class __CheckStatus(object):

    def __init__(self, params=None, *args, **kwargs):

        self._func = None
        self._emit_json = params.pop('json', False)

        if params['check'] in ['show','converge','prepare','status','reduce']:
            if 'pool' in params:
                self.poolname = pathlib.Path(params['pool'][0]).absolute()
            else:
                raise RuntimeError("Please add -f [poolname]")

        # show task status in form %
        if params['check']  == 'show':
            self.__form()

        elif params['check']  == 'converge':
            self.__form_converge()

        # update poolfile base on .status %
        elif params['check']  == 'prepare':
            self.__prepare()

        # reduce finished jobs in poolfile%
        elif params['check'] == 'reduce':
            self.__reduce()

        # update .status %
        elif params['check'] == 'status':
            self.__status()

        elif params['check']  in ['qstat','bjobs','squeue']:
            self.__qstat(params)

    @property
    def func(self):
        from jamip.abtools.vasp.setvasp import SetVasp
        from jamip.abtools.espresso.setqe import SetQE
        from jamip.abtools.cp2k.setcp2k import SetCP2K
        from jamip.abtools.gaussian.setgau import SetGaussian
        if self._func is None:
            pool = Pool().loader(self.poolname)
            func = next(iter(pool.values()))
            if not self._emit_json:
                print(type(func))
            if isinstance(func, SetVasp):
                from jamip.abtools.vasp.check import CheckStatus
            elif isinstance(func, SetQE):
                from jamip.abtools.espresso.check import CheckStatus
            if isinstance(func, SetCP2K):
                from jamip.abtools.cp2k.check import CheckStatus
            if isinstance(func, SetGaussian):
                from jamip.abtools.gaussian.check import CheckStatus
            self._func = CheckStatus
        return self._func
        
    def __status(self):
        '''
        Rebuild the. Status file
        '''
        # load tasks %
        pool = Pool().loader(self.poolname)
        for func in pool.values():
            tasks = func.tasks
            break

        root = self.poolname.parent
        success = 0
        
        with Pool.open(self.poolname, 'r') as pool:
            for key,value in pool.items():
                stdout = root/key
                status = False

                if stdout.exists():
                    status = self.func(stdout).rebuild(tasks)
                    success += 1

                value['status'] = 'C' if status is True else 'W'

        print('Total rebuild job: %s' %success)
                

    def __prepare(self):
        import logging
        
        num = 0
        # main-task-pool %
        if self.poolname.exists():
            # load tasks %
            pool = Pool().loader(self.poolname)
            for func in pool.values():
                tasks = func.tasks
                break
         
            root = self.poolname.parent
         
            with Pool.open(self.poolname, 'w') as pool:
                for path,value in pool.items():
                    stdout = root/path
                    job_status = False 
         
                    if stdout.exists():
                        job_status = True 
                        status = self.func.load_status(stdout) 
                        for key in tasks:
                            if not status.get(key, False) or status[key]['status'] is False:
                                job_status = False

                    job_status = 'C' if job_status else 'W'
                    if value['status'] != job_status:
                        value['status'] = job_status
                        pool[path] = value
                        logging.info(f'Update Job: {path}')
                        num +=1
        else:
            #sub-task-pool
            print("[Warning] - Get task list failed. Try update state file only.")

            # load pool %
            with Pool.open(self.poolname, 'w') as pool:
                for key,value in pool.items():
                    if value['status'] == 'C':
                        pool.update_status(key, status='W')
                        num += 1

        print(f'Total rebuild job: {num}')

    def __reduce(self):
        '''
        Rebuild the. Status file
        '''
        import logging

        # main-task-pool %
        if self.poolname.exists():

            reduced = []
            with Pool.open(self.poolname, 'w') as pool:
                for path,value in pool.items():
                    if value['status'] == 'C':
                        reduced.append(path)
                for path in reduced:
                    del pool[path]

                logging.info('Reduce Job: %d' %len(reduced))


    def __qstat(self,params):
        import json
        from jamip.compute.manager import TaskManager
        
        cwd = str(pathlib.Path.cwd().resolve())
        order = params['check']

        if order == 'qstat':
            tm = TaskManager('pbs') 
        elif order == 'bjobs':
            tm = TaskManager('lsf') 
        elif order == 'squeue':
            tm = TaskManager('slurm') 

        jobs = []

        if 'pool' in params:
            for jobid in params['pool']:
                jobid = str(jobid)
                if jobid and jobid.isdigit():
                    abspath = tm.get_task_by_id(jobid)
                    entry = {'id': jobid, 'path': abspath}
                    if len(abspath) > 30 and abspath.startswith(cwd):
                        entry['path_relative_to_cwd'] = abspath[len(cwd)+1:]
                    jobs.append(entry)
        else:
            jobdf = tm.get_task_by_user()
            if len(jobdf) != 0:
                for job in jobdf.itertuples():
                    abspath = job.path
                    entry = {
                        'id': str(job.id),
                        'status': str(job.status),
                        'path': abspath,
                    }
                    if len(abspath) > 30 and abspath.startswith(cwd):
                        entry['path_relative_to_cwd'] = abspath[len(cwd)+1:]
                    jobs.append(entry)

        if self._emit_json:
            print(json.dumps({
                'ok': True,
                'check': order,
                'cwd': cwd,
                'jobs': jobs,
                'job_count': len(jobs),
            }, ensure_ascii=False, indent=2, default=str))
            return

        for entry in jobs:
            jid = entry['id']
            abspath = entry['path']
            if 'path_relative_to_cwd' in entry:
                print(jid, ':', entry['path_relative_to_cwd'])
            else:
                print(jid, ':', abspath)
        
    def __form_converge(self):
        """
        对任务池中已提交但未完成的任务进行检查，获取其详细状态
        1. 任务计算路径
        2. 当前任务状态(运行中&已结束)
        3. 当前任务进度(最后计算的子目录,基于jamip.log)
        4. 目前的能量收敛值
        5. 目前的结构与原始结构的差异 (原始结构来自info)
        6. 修改建议
        """
        from jamip.utils.views import shellform
        from jamip.compute.manager import TaskManager
        from jamip.analysis.vasp.outcar import GrepOutcar
        from jamip.structure.convert import read_structure_from_hdf5
        from jamip.structure import read
        import pandas as pd
        import pathlib
        import re

        data = []
        columns = ['path','status','task','E0','dE','cell','note']
        # get manager %
        #manager = Cluster('./').manager.lower()
        tm = TaskManager.from_yaml('./')
        jobdf = tm.get_task_by_user()
        jobdf = jobdf[jobdf['status']=='R']

        # load tasks %
        pool = Pool().loader(self.poolname)
        for func in pool.values():
            tasks = [key for key in func.tasks.keys()]
            break

        # load pool %
        with Pool.open(self.poolname, 'r') as pool:
            for key,value in pool.items():
                outdir = pathlib.Path(key)

                job_status = 'C'
                if outdir.absolute() in jobdf['path'].values:
                    job_status = 'R'

                task = None
                taskdir = None
                logpath = None
                E0 = None
                dE = None
                dS = None
                note = None
                tasks = ['relax']

                # get unfinish task %
                status = self.func.load_status(outdir)
                for key in tasks:
                    if key not in status:
                        task = key
                        break
                    elif status[key]['status'] != True:
                        task = key
                        taskdir = status[key]['path'] 
                        break

                if task == None: continue

                # get task path % 
                logfile = outdir/'jamip.log'
                retask = re.compile(r"file='(\S+/%s/\S+)'" %task)
                if logfile.exists():
                    with open(logfile, 'r') as f:
                        for line in f:
                            result = retask.search(line)
                            if result:
                                logpath = pathlib.Path(result.groups()[0])
                                taskdir = logpath.parent.relative_to(outdir.absolute()) 

                # get energy and cell (if relax) %
                if taskdir != None and (outdir/taskdir).exists():

                    # get energy %
                    # 考虑能量收敛有几个角度，
                    # 1. 总能的值总归要是一个负值
                    # 2. 每个离子步中的电子步dE在不断减小，且可以减小到收敛值
                    # oszicar的三列为 E, E0, dE
                    oszicar = outdir/taskdir/'OSZICAR'
                    if oszicar.exists():
                        oszicar = GrepOutcar().oszicar(str(oszicar.parent))
                        if len(oszicar):
                            E0 = '%.2E' %oszicar[-1,1]
                            dE = '%.2E' %oszicar[-1,2]
                    
                    # 仅在优化中需要考虑结构是否崩溃，可以通过比较结构差异或体积确定
                    # 1. 计算结构中的平均原子位移
                    # 2. 比较优化前后的体积
                    # 其中如果任务是优化，需要读取初始结构(来自info.hdf5)
                    # 如果任务是其他，只需要比较当前目录的CONTCAR和POSCAR即可 
                    contcar = outdir/taskdir/'CONTCAR'
                    if contcar.exists() and contcar.stat().st_size > 0:
                        s = read(contcar)

                        s0 = None
                        if re.match(r'relax/S\d+', str(taskdir)):
                            s0 = read_structure_from_hdf5(str(outdir/'info.hdf5'), key='structure/raw')

                        if s0 is None:
                            poscar = outdir/taskdir/'POSCAR'
                            s0 = read(poscar)

                        dS = '%.2f' %(s.volume / s0.volume)

                    #force = round(GrepOutcar().max_force(join(root,path)), 3)
                        
                    # try get logfile %
                    logpath = None
                    if logpath == None:
                        logs = (outdir / taskdir).glob('*.log') 
                        for log in logs:
                            if log.stat().st_size > 0:
                                logpath = log
                                break

                # get energy and cell (if relax) %
                else:
                    taskdir = task
                    note = 'uncalculated'

                row = [outdir, job_status, taskdir, E0, dE, dS, note] 
                # get status %

                # get last relax path %
                #status = self.func.load_relax_status(outdir)
                #if status == None: continue
                #data.update(status)
                #data['last'] = os.path.basename(data['last'])
                data.append(row)

        if len(data) > 0:
            data = pd.DataFrame(data, columns=columns)
            shellform(data)
        else:
            print("All jobs converge.")

    def __form(self):
        from jamip.utils.views import shellform 
        import pandas as pd
        import json

        def _coerce_job_id(raw):
            if isinstance(raw, int):
                return raw
            if isinstance(raw, str):
                try:
                    return int(raw.strip())
                except ValueError:
                    return raw
            return raw

        data = []
        columns = None

        if self.poolname.exists():
            pool = Pool().loader(self.poolname)
            for func in pool.values():
                tasks = [key for key in func.tasks.keys()]
                break
            if not self._emit_json:
                print(tasks)

            columns = ['id','prior','status'] + tasks + ['path']
            root = self.poolname.parent

            with Pool.open(self.poolname, 'r') as pool:
                for path_key, value in pool.items():
                    outdir = path_key
                    row = [_coerce_job_id(value['id']), value['prior'], value['status']]

                    if value['status'] in ['C','R'] or value['prior'] < 9:
                        status = self.func.load_status(root/outdir)
                        for task_name in tasks:
                            if task_name in status:
                                row.append(status[task_name]['status'])
                            else:
                                row.append(None)
                    else:
                        row += [None] * len(tasks)

                    row.append(outdir)
                    data.append(row)

        else:
            if not self._emit_json:
                print("[Warning] - Read Data file failed. Try read state file only.")
                print(f"  → 未找到任务池数据文件（应与 -f 同名的 *.pool）:")
                print(f"     {self.poolname.resolve()}")
                print("  → 请先在当前目录成功执行: jp -r prepare -f <同名.pool>")
                print("  → 若 prepare 中途报错退出（例如缺少 pots/Si/POTCAR 或未设置 JAMIP_PAW_PBE），不会生成该文件。")

            columns = ['row', 'prior', 'status']
            with Pool.open(self.poolname, 'r') as pool:
                for path_key, value in pool.items():
                    row = [path_key, value['prior'], value['status']]
                    data.append(row)

        if self._emit_json:
            workflow_steps = None
            if self.poolname.exists() and columns and 'path' in columns:
                workflow_steps = [c for c in columns if c not in ('id', 'prior', 'status', 'path')]

            rows_out = []
            for row in data:
                rowd = dict(zip(columns, row))
                if workflow_steps:
                    for step in workflow_steps:
                        rowd[step] = rowd.get(step) is True
                rows_out.append(rowd)

            payload = {
                'ok': True,
                'check': 'show',
                'pool_file': str(self.poolname.resolve()),
                'pool_data_file_exists': self.poolname.exists(),
                'columns': columns,
                'rows': rows_out,
                'row_count': len(rows_out),
            }
            if not self.poolname.exists():
                payload['hint_zh'] = (
                    '未找到与 -f 同名的 pickle 任务池文件；请先在同一目录成功运行 jp -r prepare。'
                    'prepare 若因赝势等错误退出则不会生成该文件，show 只会显示空表。'
                )
            if workflow_steps is not None:
                payload['workflow_steps'] = workflow_steps
                payload['hint_zh'] = (
                    '每条 structure 对应 workflow 子步骤是否完成；'
                    'pool_status 为池级调度状态（英文首字母）：'
                    'W=Waiting 等待投递/排队（不是「完成」）；'
                    'R=Running 运行中；'
                    'C=Complete 池里本条已标记完成。'
                    'steps / rows 里各步布尔：true=该步已成功，false=尚未成功'
                )
                structures = []
                for rowd in rows_out:
                    steps_bool = {step: rowd[step] for step in workflow_steps}
                    done_n = sum(steps_bool.values())
                    total = len(workflow_steps)
                    structures.append({
                        'output_path': rowd['path'],
                        'job_id': rowd['id'],
                        'prior': rowd['prior'],
                        'pool_status': rowd['status'],
                        'steps': steps_bool,
                        'progress': f'{done_n}/{total}',
                    })
                payload['structures'] = structures
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            return

        data = pd.DataFrame(data, columns=columns)
        shellform(data)

# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
from time import sleep
import numpy as np
from src.utils.variables import state_of_calculation

class Submit:
    def __init__(self, pool):
        """
        submit jobs in pool.
        
        Arguments:
            pool: pool's object.
        """
        self._pool=pool
        
    @property
    def pool(self):
        return self._pool
        
    def run(self, maxJob, **kwargs):
        """
        submit jobs
        
        Arguments:
            maxJob: maximum of parallel jobs.
            
            path_of_flag (default: current path): path of flag file (over) for stopping the submit process.
            time_of_sleep (default=5): interval time for checking and submitting jobs.
        """
        
        path_of_flag=os.getcwd()
        if 'path_of_flag' in kwargs:
            path_of_flag=kwargs['path_of_flag']
        time_of_sleep=5
        if 'time_of_sleep' in kwargs:
            time_of_sleep=kwargs['time_of_sleep']
        
        state=True
        while(state):
            # check
            if os.path.exists('{}/over'.format(path_of_flag)):
                state=False
                os.remove('{}/over'.format(path_of_flag))
                break
            
# =============================================================================
#             prepare_jobs=self.pool.get_prepare_jobs()
#             calculating_jobs=self.pool.get_calculating_jobs()
# =============================================================================
            prepare_jobs=self.pool.get_jobs(state_of_calculation=state_of_calculation.prepare)
            calculating_jobs=self.pool.get_jobs(state_of_calculation=state_of_calculation.calculating)
            print('prepare: {}'.format(prepare_jobs))
            print('calculating: {}'.format(calculating_jobs))

            num_prepare_jobs=len(prepare_jobs)
            num_calculating_jobs=len(calculating_jobs)
            if num_prepare_jobs == 0:
                state=False
            elif num_calculating_jobs < maxJob:
                num_allowed_jobs=maxJob-num_calculating_jobs # number of jobs allowed to be submitted
                num_sumbit_jobs=np.min([num_allowed_jobs, num_prepare_jobs]) # number of jobs to prepare for submission
                
                for i in range(0, num_sumbit_jobs):
                    job=prepare_jobs[i]
                    job.run()
            
            sleep(time_of_sleep)
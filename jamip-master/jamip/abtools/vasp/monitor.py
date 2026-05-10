import threading
import logging
import psutil
import socket
import time
import os
from jamip.utils.logger import children_info, load_yaml

def find_mpirun_process(p, mpi='mpirun'):
    for sh in p.children():
        try:
            if len(sh.children()) == 0 :continue
            for child in sh.children():
                if child.name() == mpi:
                    return child
        except:
            pass

class VASP_Thread(threading.Thread):

    def __init__(self, func, args=()):
        '''
        args: ( VaspFlow, incar, stdout )
        '''
        super(VASP_Thread,self).__init__()
        self.func = func
        self.args = args
        self.stdout = args[-1]
        self.mpi = args[0].cluster.mpi.split()[0]
        self.logfile = None
        self.pointer = 0
        self.error = None
        # load errormap %
        errors = load_yaml(os.environ['HOME']+'/.jamip/env/error.yaml')
        self.error_map = errors if errors else {}

    def run(self):
        logging.debug('Task start')
        self.result = self.func(*self.args)
        logging.debug('Task end')

    def get_result(self):
        try:
            return self.result 
        except Exception:
            return None

    def read_logs(self):
        if self.logfile != None:
            logfile = self.logfile 
        else:
            logfile = os.path.join(self.stdout, 'vasp.log')

        with open(logfile, "r") as f:
            f.seek(self.pointer,0)
            lines = f.readlines()
            for line in lines:
                for key,value in self.error_map.items():
                    if value[0] in str(line):
                       self.error = key
                       break
            self.pointer = f.tell()

        if self.error != None:
            try:
                process = psutil.Process(self.pid)
                process.terminate()
            except:
                pass

    def find_logs(self, p):
        child = find_mpirun_process(p, self.mpi)
        if child is not None:
            self.pid = child.pid
            self.logfile = child.open_files()[0].path
            self.pointer = 0
            logging.info("Monitor: [ pid='%s', name='%s', file='%s']" %(child.pid, child.name(), self.logfile))

    def debug(self, incar, *args, **kwargs):
        import re
        params = self.error_map[self.error][1]
        if params == None:
            logging.info("No debug options are available, exit.")
            raise SystemExit("Exit vasp calculation for Error %s" %self.error)

        for key,value in params.items():
            if value is None or isinstance(value,(int,float,bool)):
                incar[key] = value
            elif isinstance(value, (str,unicode)):
                if 'true' in value.lower():
                    incar[key] = True
                elif 'false' in value.lower():
                    incar[key] = False
                elif '+' in value or '-' in value or '*' in value or '/' in value:
                    t = re.findall(r'[A-Za-z]+',value)
                    for tag in t:
                        value = value.replace(tag,str(jg.grep(tag.lower())),1)
                    incar[key] = eval(value)

        return params


def Monitor(func):
    def warp(self, incar, stdout, **kwargs):

        # search running vasp %
        # CPUcheck.write_children_status('job start.')

        # run vasp thread %
        taskname = os.path.relpath(stdout,self.rootdir)
        monitor=VASP_Thread(func, args=[self, incar, stdout])
        monitor.start()

        sleep = 1
        pointer = -1
        maxsleep = self.cluster.maxsleep
        p = psutil.Process(os.getpid())

        while threading.activeCount() > 1:

            # wait subprocess start %
            if monitor.logfile is None:
                monitor.find_logs(p)
                if sleep < 10:
                    time.sleep(5)
                    sleep += 1
                else:
                    logging.error('exit for timeout')
                    break

            # read logs durning subprocess active %
            elif sleep < maxsleep:
                logging.debug('sleep = %ss' %sleep)
                logging.debug('Error: %s' %monitor.error)
                time.sleep(min(sleep,60))
                monitor.read_logs()
                if pointer != monitor.pointer:
                    pointer = monitor.pointer
                    sleep = int(sleep/10)+1
                else:
                    sleep += min(sleep, 60)
          
            # overtime %
            else:
                logging.warning('Warning: sleep overwrite! time=%s' %sleep)
                break

        # finally read vasplog after subprocess finish %
        monitor.read_logs()

        # Analyze subprogress status after finish%
        monitor.join(60)
        if monitor.get_result() != None:
            stdout = monitor.get_result()
            logging.info("Monitor: %s join" %taskname)
        else:
            os.chdir(self.rootdir)
            if sleep >= 1000:
                logging.info("Monitor: %s timeout" %taskname)
                children_info()
            elif monitor.error is not None:
                logging.error("Monitor: %s error for %s" %(taskname, monitor.error))
            # kill mpirun %
            child = find_mpirun_process(p)
            if child is not None:
                logging.warning("Residual process: %s" %child)
                grandchildren = child.children(recursive=True)
                child.terminate()
                # kill child's children
                for c in grandchildren():
                    if c.parent().pid == 1:
                        c.terminate()


        # debug %
        if monitor.error is not None and incar.state != "D":
            incar.state = "D"
            if monitor.logfile and os.path.exists(monitor.logfile):
                os.rename(monitor.logfile, os.path.join(stdout, 'error.log'))
            monitor.debug(incar, monitor.error, stdout)
            logging.info("Monitor: %s debug" %taskname)
            self.calculator(incar.name, stdout)
            # move dictorys %
            #debugout = os.path.join(self.rootdir,'debug',taskname.replace('/','_'))
            #os.popen("mv {0} {1}".format(stdout,debugout)).readline()
            #os.popen("mv {0} {1}".format(debugin,stdout)).readline()

        return stdout
    return warp

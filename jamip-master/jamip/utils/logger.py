import ruamel.yaml
import logging
import os
import numpy as np


logging.basicConfig(format='[%(levelname)s] - %(message)s', level=logging.ERROR)
os.environ['NUMEXPR_MAX_THREADS'] = '4'

def run_logger(path=None):
    import logging.config
    if path is None: 
        home = os.environ.get('JAMIP_HOME', f"{os.environ['HOME']}/.jamip")
        path = os.path.join(home, 'env', 'logger.yaml')
    try:
        logconf = load_yaml(path)
        logging.config.dictConfig(logconf)
        logging.debug("Logging read configfile success.")
    except:
        logging.error("Warning! Load logging configuration failed! ")

def add_logger(path='debug.log', name=None, level=logging.INFO):
    fhlr = logging.FileHandler(path, mode='w')
    fhlr.setLevel(level)
    fhlr.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger = logging.getLogger(__name__)
    logger.addHandler(fhlr)
    return logger

def full_path(path:str, label='Path'):

    from os.path import exists, expanduser, abspath
    fpath = abspath(expanduser(path))

    if not exists(fpath):
        logging.error("File not exists : %s" %path)
        exit()

    return fpath

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    import numpy as np
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def load_yaml(path, strict: bool = True):

    path = os.fspath(path)
    data = None
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                yml = ruamel.yaml.YAML(typ='safe', pure=True)
                data = yml.load(f)
                logging.debug("Load YAML : %s" %path)
        except Exception as e:
            logging.error("YAML Syntax Error : %s" %path)
            logging.error("YAML Detail : %s" %repr(e))
            if strict:
                exit()
            return None

    return data

def convert_numpy(obj):

    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(item) for item in obj]
    else:
        return obj    

def dump_yaml(data, path:str, default=False, clean=False):

    if clean:
        data = convert_numpy(data)

    with open(path, 'w') as f:
        yml = ruamel.yaml.YAML()
        yml.indent(sequence=3)
        if default is False: 
            yml.dump(data, f)
        else:
            yml.default_flow_style = False
            yml.dump(data, f)

def load_hdf5(key:str, path='info.hdf5'):
    import h5py

    data = None
    try:
        with h5py.File(path, mode="r") as h5: 
            if key in h5:
                if key.startswith('/structure') or key.startswith('structure'):
                    group = h5[key]
                    data = (group['lattice'][()], group['positions'][()], group['elements'][()])
                else:
                    data = {}
                    for attr in  h5[key].attrs:
                        data[attr] = h5[key].attrs[attr]
    except:
        pass
    
    return data


def dump_hdf5(key:str, value, path='info.hdf5'):
    from jamip.structure import Structure
    from jamip.abtools.base.kpoints import BandPath
    import h5py

    os.environ['HDF5_USE_FILE_LOCKING'] = "FALSE"
    with h5py.File(path, mode="a") as h5:

        if key in h5: del h5[key]

        if isinstance(value, Structure): 
            g = h5.create_group(key)
            g['lattice'] = value.lattice
            g['elements'] = value.get_elements()
            g['positions'] = value.get_positions(type='direct')
        
        elif isinstance(value, BandPath): 
            g = h5.create_group(key)
            g['paths'] = [i.symbol for i in value.sites]
            g['kpoints'] = [i.position for i in value.sites]
            g['numbers'] = value.numbers

        elif isinstance(value, tuple): 
            lattice, positions, elements = value
            g = h5.create_group(key)
            g['lattice'] = lattice
            g['elements'] = elements
            g['positions'] = positions

        elif isinstance(value, dict):
            d = h5.create_dataset(key, dtype="f")
            for k,v in value.items():
                d.attrs[k] = v

def cluster_info(ncore=20, cmd='mpirun'):

    import psutil 
    import socket

    ncpu = psutil.cpu_count()
    pcpu = psutil.cpu_percent()
    pmem = psutil.virtual_memory().percent
    host = socket.gethostname()

    logging.info('Cluster: [host=%s, NCPU=%s, PCPU=%s%%, PMEM=%s%%]' %(host,ncpu,pcpu,pmem))
    # check %
    if ncpu != ncore:
        logging.warning("The actual CPU cores not match the set value!")
    if pcpu >  90:
        logging.warning("Excessive cpu usage!")
    if pmem >  90:
        logging.warning("Excessive memory usage!")
    # process check %
    vasppids = []
    for p in psutil.process_iter():
        if cmd in p.name():
            logging.warning("%s job is running, job will exit." %cmd)
            #try:
            #    p.terminate()
            #except:
            #    logging.error("kill mpirun failed.\n")
        elif 'vasp' in p.name():
            if p.parent().pid == 1:
                logging.error('pid %s is an orphan vasp process.' %p.pid)
            else:
                vasppids.append(p.pid)

def children_info(text=''):

    import psutil
    p = psutil.Process(os.getpid())
    logging.info("Thread : %s" %p.num_threads())
    for i,sh in enumerate(p.children()):
        logging.info("Thread-sh_%d : %s" %(i,sh.children()))
    

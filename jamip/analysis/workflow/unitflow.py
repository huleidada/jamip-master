from jamip.structure import read, Structure
from jamip.structure.dimension import DimensionAnalysis, AtomsError, Unit
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import spglib
import pathlib
import logging
import shutil
import json
import six
from dataclasses import dataclass
import threading

@dataclass
class flowvars:
    """
    global variables
    emin, emax: energy range
    limit: energy range for dos plot
    scissor: energy shift for band/dos above fermi level
    title: title of the plot
    xlabel: xlabel of the plot
    ylabel: ylabel of the plot
    """
    source: str
    output: str
    debug: bool=False
    level: str='info'
    logfile: str='run.log'
    logmode: str='a'

class globalvar:
    """
    global variables for different plot types
    """
    unitflow   = flowvars(source=None, output=None, debug=False, level=logging.DEBUG, logfile='unit.log', logmode='a')
    icsdflow  =  flowvars(source=None, output=None, debug=False, level=logging.DEBUG, logfile='icsd.log', logmode='a')
    df = pd.DataFrame()
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = object.__new__(cls)  
        return cls._instance

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def get_dimension(cif):

    source = pathlib.Path(globalvar.unitflow.source)
    savedir = pathlib.Path(globalvar.unitflow.output)

    s = read(source / cif)
    stdcell = spglib.refine_cell(s.to_cell(), symprec=1e-1)
    s = Structure.from_cell(stdcell)

    dim = DimensionAnalysis(s)
    dim.cutoff = 1.1
    dim.set_valence()
    dim.set_bonding()
    ndim = dim.search_cutoff()
    data = {'cif': cif, 'dim':ndim}
    if ndim != 2:
        return data
        
    # 仅保留结构单元,不保存额外信息 
    try:
        dim.get_2d_units()
        units, layersites, layershifts = dim.get_interface()
    except AtomsError:
        return data
    except Exception as err:
        with open("error.log", 'a') as f:
            f.write(cif+'\n')
        value = sys.exc_info()
        six.reraise(*value)
        print(self.dataset[idx].name)
        print(err)
        exit()

    if globalvar.unitflow.debug == True:
        pass

    unitlist = []
    for i,unit in enumerate(units):
        if unit.dim == 2:
            idx = cif.split('.')[0]
            write(unit.stdcell, savedir/('%s-%d.vasp' %(idx, i)))
            unitlist.append(unit.to_dict())
            #print(unit)
            #unit.check_pointgroup_symmetry()
            #print("Check pointgroup symmetry success!")
            #unit.check_unique_pointgroup_symmetry()
            #print("Check unique pointgroup symmetry success!")
    data['unit'] = unitlist

    unit_indices = []
    operations = []
    distances = []
    arearates = []
    spacings = []
    for lsf in layershifts:
        idx = lsf.index[0] 
        unit_indices.append(layersites[idx].index)
        operations.append(layersites[idx].symop)
        distances.append(lsf.distance)
        arearates.append(lsf.distance_area_rate) 
        spacings.append(lsf.spacing_in_cartesian) 
    
    data['unit_indices'] = unit_indices
    data['operations'] = operations
    data['distances'] = distances 
    data['arearates'] = arearates 
    data['spacings'] = spacings 

    logging.info("Finsih | "+cif)

    return data

def get_dimension_from_units(cif):

    df = globalvar.df

    s = read(cif)
    idx = re.match('ICSD-(\d+)-\d.vasp', cif.name)
    key = 'ICSD-%s.cif' %idx.groups()[0]
    # get charges %
    valence = json.loads(df[key]['valence'])
    unit = Unit.from_structure(s)
    unit.set_valence(valence)
    total_val = unit.total_valence
    #if min(unit.valences.values()) > 0: total_val = 0

    # 获得表面原子的种类，上下表面各一个，深度限制为0.2 nm
    unit.get_edge_indices(charge=True)

    print(unit.sites['u1'])
    print(unit.sites['u2'])
    print(unit.sites['d1'])
    print(unit.sites['d2'])

    assert unit.sites['u1'].specie == unit.sites['u2'].specie
 
    # DATA: FORMULA
    data = unit.to_dict()
    data['total_charge'] = unit.total_charge
    data['total_charge'] = unit.mean_multiplicity,
    

class FlowManager:

    def __init__(self, func:str):
        if func == 'unit':
            self.var = globalvar.unitflow
            self.func = get_dimension
        elif func == 'icsd':
            self.var = globalvar.icsdflow
        else:
            raise KeyError("Unknown function")

        if self.var.source is None:
            raise OSError("Miss source directory.")

        if self.var.logfile != None:
            logging.basicConfig(format='[%(levelname)s] - %(message)s', 
                                level=self.var.level,
                                filename=self.var.logfile,
                                filemode=self.var.logmode)

    def load_logger(self):

        finishs = []
        errors = []
        with open(self.var.logfile, 'r') as f:
            for line in f:
                result = line.split()
                if result[1] == 'Finish':
                    finishs.append(result[2])
                elif result[1] == 'Error':
                    errors.append(result[2])

        self.errors = errors
        self.finishs = finishs                   

    def mpirun(self, cifs, max_workers=30):
        jsons = []
        pbar = tqdm(total=len(cifs))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for data in executor.map(self.func, cifs):
                jsons.append(data)
                pbar.update(1)
        pbar.close()
        return jsons

    def test(self, cifs, row=1):
        if isisntance(cifs, str):
            return self.func(cifs, debug=True)
        else:
            jsons = []
            for i in cifs:
                jsons.append(self.func(i, debug=True))
                if len(jsons) >= row:
                    break 
            return jsons
               

    def run(self, cifs):
        import time
        jsons = []
        pbar = tqdm(total=len(cifs))
        for cif in cifs:
            t0 = time.time()
            data = self.func(cif, debug=True)
            t1 = time.time()
            print(cif, t1-t0)
            jsons.append(data)
            pbar.update(1)
 
        pbar.close()
        return jsons

if __name__ == "__main__":

    from jamip.analysis.workflow import FlowManager, globalvar, default_dump
    import pandas as pd
    import cProfile
    import pathlib
    import json

    # initialize parameters 
    globalvar.unitflow.source = '/public/home/kzhou/Src/icsd2022/'
    globalvar.unitflow.output = pathlib.Path('./units')
    globalvar.unitflow.logmode = 'a'
    fm = FlowManager(func='unit')
    fm.load_logger()

    # load input  
    csvfile = 'dim-jamip.csv'
    df = pd.read_csv(csvfile)
    df2 = df[(df['dim']==2) & (df['spacegroup']>=75)]
    cifs = []
    for cif in df2['cif']:
        if cif not in fm.finishs:
            cifs.append(cif)

    # test
    jsons = fm.test(cifs, row=3)

    # save json 
    jsons = fm.mpirun(cifs, max_workers=30)
    with open('unit.json', 'a') as f:
        for data in jsons:
            if data != None:
                print(data)
                f.write(json.dumps(data, default=default_dump))
                f.write('\n')


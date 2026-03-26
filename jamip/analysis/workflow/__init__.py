from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import numpy as np
import spglib
import pathlib
import logging
import shutil
import json
import six
import sys
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
    unitflow   = flowvars(source=None, output=None, debug=False, level=logging.INFO, logfile='unit.log', logmode='a')
    icsdflow  =  flowvars(source=None, output=None, debug=False, level=logging.INFO, logfile='icsd.log', logmode='a')
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

def get_dimension(cif, write_unit=True):
    from jamip.structure import read, write, Structure
    from jamip.structure.dimension import DimensionAnalysis, AtomsError

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
            if write_unit:
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

def get_unit_data(cif):
    from jamip.structure import read, write, Structure
    from jamip.structure.dimension import Unit
    import re

    source = pathlib.Path(globalvar.unitflow.source)
    data = {'cif': cif.name}

    # get charges %
    idx = re.match('ICSD-(\d+)-\d.vasp', cif.name)
    key = 'ICSD-%s.cif' %idx.groups()[0]
    valence = json.loads(globalvar.unitflow.dataframe[key]['valence'])
    
    s = read(source / cif)
    unit = Unit.from_structure(s)
    unit.set_valence(valence)

    #total_val = unit.total_val
    #if min(unit.valence.values()) > 0: total_val = 0
    if globalvar.unitflow.debug == True:
        #print(unit)
        #unit.check_pointgroup_symmetry()
        #print("Check pointgroup symmetry success!")
        #unit.check_unique_pointgroup_symmetry()
        #print("Check unique pointgroup symmetry success!")
        pass

    unit.get_edge_indices()
    data.update(unit.to_dict())

    u1 = unit.sites['u1']
    d1 = unit.sites['d1']

    # 界面原子判据
    # 1. 元素种类一致 
    surface = True
    if u1.specie != d1.specie:
        surface = False
    # 2. 价态与总价态一致 
    if surface and valence[u1.specie] * unit.total_charge <= 0:
        surface = False
    # 3. 次近邻原子未明显高出表面
    if surface and unit.sites['d2'] != None: 
        if d1.cartesian - unit.sites['d2'].cartesian > 0.2:
            surface = False
    if surface and unit.sites['u2'] != None: 
        if unit.sites['u2'].cartesian - u1.cartesian > 0.2:
            surface = False
    data['valid'] = surface

    logging.info("Finsih | %s" %cif.name)

    return data


class FlowManager:

    def __init__(self, func:str):
        if func == 'unit':
            self.var = globalvar.unitflow
            self.func = get_dimension
        elif func == 'unitdir':
            self.var = globalvar.unitflow
            self.func = get_unit_data
        elif func == 'icsd':
            self.var = globalvar.icsdflow
        else:
            raise KeyError("Unknown function")

        if self.var.source is None:
            raise OSError("Miss source directory.")

        if self.var.output is None:
            self.var.output = pathlib.Path.cwd()
        else:
            self.var.output = pathlib.Path(self.var.output)
        if not self.var.output.exists():
            self.var.output.mkdir()

        if self.var.logfile != None:
            logging.basicConfig(format='[%(levelname)s] - %(message)s', 
                                level=self.var.level,
                                filename=self.var.logfile,
                                filemode=self.var.logmode)

    def load_logger(self):

        finishs = []
        errors = []
        logfile = pathlib.Path(self.var.logfile)
        if logfile.exists():
            with open(logfile, 'r') as f:
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
        if isinstance(cifs, str):
            return self.func(cifs)
        else:
            jsons = []
            for i in cifs:
                jsons.append(self.func(i))
                if len(jsons) >= row:
                    break 
            return jsons
               
    def run(self, cifs):
        jsons = []
        pbar = tqdm(total=len(cifs))
        for cif in cifs:
            data = self.func(cif)
            jsons.append(data)
            pbar.update(1)
 
        pbar.close()
        return jsons

if __name__ == "__main__":

    # add dimension
    exit()

    # dim analysis
    jsons = run(get_dimension, df2['cif'])#.values[:10])
    with open('unit.json', 'a') as f:
        for data in jsons:
            if data != None:
                print(data)
                f.write(json.dumps(data, default=default_dump))
                f.write('\n')

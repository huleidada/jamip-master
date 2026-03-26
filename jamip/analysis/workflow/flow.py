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

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class FlowManager:

    def __init__(self, func, logger=None):
        self.func = func

 
        if logger == None:
            logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(format='[%(levelname)s] - %(message)s', 
                                level=self.var.level,
                                filename=self.var.logfile,
                                filemode=self.var.logmode)

    def initialize(self, **kwargs):
        self.func = self.func(**kwargs)

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
        gene = self.func.generator(cifs)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for data in executor.map(self.func.run, gene):
                if isinstance(data, list):
                    jsons.extend(data)
                else:
                    jsons.append(data)
                pbar.update(1)
        pbar.close()
        return jsons

    def test(self, cifs, row=1):
        jsons = []
        pbar = tqdm(total=row)
        gene = self.func.generator(cifs)
        for dat in gene:
            data = self.func.run(dat)
            if isinstance(data, list):
                jsons.extend(data)
            else:
                jsons.append(data)
            pbar.update(1)
            if len(jsons) >= row: 
                break
 
        pbar.close()
        return jsons
               
    def run(self, cifs):
        import time
        jsons = []
        #pbar = tqdm(total=len(cifs))
        gene = self.func.generator(cifs)
        for dat in gene:
            try:
                t0 = time.time()
                data = self.func.run(dat)
                t1 = time.time()
                if isinstance(data, list):
                    jsons.extend(data)
                else:
                    jsons.append(data)
            except:
                print('error')
            #print(t1-t0)
        #    pbar.update(1)
 
        #pbar.close()
        return jsons

if __name__ == "__main__":


    # dim analysis
    jsons = run(get_dimension, df2['cif'])#.values[:10])
    with open('unit.json', 'a') as f:
        for data in jsons:
            if data != None:
                print(data)
                f.write(json.dumps(data, default=default_dump))
                f.write('\n')

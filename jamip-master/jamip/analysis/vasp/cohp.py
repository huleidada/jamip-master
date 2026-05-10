import os
import re
import numpy as np
import pandas as pd
from jamip.analysis.base import Finder
from collections import defaultdict

tolatex = lambda match: '%s{%s}' %(match.group()[0], match.group()[1:])

def orbit2latex(string):
    if string == '':
        return ''
    string = string.strip('[]')
    string = re.sub(r'(\^\d)',tolatex,string)
    string = re.sub(r'(_\S*)',tolatex,string)
    return '$_{%s}$' %string 

class COHPFinder(Finder):

    def __init__(self,path):
        self.task = 'cohp'
        self.soft = 'vasp'
        self.file = None
        self.stdin = path

    def read_lobsterin(self,path):
        """
        Read lobsterin.lobster
        """
        with open(os.path.join(path,'lobsterin.lobster'),'r') as f:
            atoms = {}
            for line in f:
                if line.startswith('basisfunctions'):
                    result = line.split()
                    # Mo 4p 4d 5s
                    atoms[result[1]] = result[2:]
        return atoms

    def read_icohp(self,path,dtype='p'):
        return self.read_coopcar(os.path.join(path,'COHPCAR.lobster'), dtype=dtype, offset=2)

    def read_icoop(self,path,dtype='p'):
        return self.read_coopcar(os.path.join(path,'COOPCAR.lobster'), dtype=dtype, offset=2)

    def read_cohp(self,path,dtype='p'):
        return self.read_coopcar(os.path.join(path,'COHPCAR.lobster'), dtype=dtype)

    def read_coop(self,path,dtype='p'):
        return self.read_coopcar(os.path.join(path,'COOPCAR.lobster'), dtype=dtype)

    def read_coopcar(self,path,dtype='m',offset=1):
        """
        Read COOPCAR.lobster or COHPCAR.lobster
        """
        with open(path,'r') as f:
             f.readline()
             row, _, points, emin, emax, efermi = f.readline().split()
             labels = []
             for i in range(int(row)):
                 labels.append(f.readline())                 
             coop = []
             for i in range(0,int(points)):
                 coop.append(f.readline().split())
             coop = np.array(coop,dtype=float)
             
        if dtype == 'm':
            pattern = re.compile(r'([A-Z][a-z]?)[0-9]+(\[\S*\])?->([A-Z][a-z]?)[0-9]+(\[\S*\])')
        elif dtype == 'p':
            pattern = re.compile(r'([A-Z][a-z]?)[0-9]+\[(\d[spdf])\S*\]->([A-Z][a-z]?)[0-9]+\[(\d[spdf])\S*\]')
        elif dtype == 't':
            pattern = re.compile(r'([A-Z][a-z]?)[0-9]+->([A-Z][a-z]?)[0-9]+')

        data = {}
        ndata = defaultdict(int)
        data['energy'] = coop[:,0]
        for i,label in enumerate(labels):

            result = pattern.findall(label)
            if len(result) == 0: continue
            result = result[0]    
            # label = '%s%s - %s%s' %(result[0],orbit2latex(result[1]),result[2],orbit2latex(result[3]))
            if dtype == 't':
                label = '%s->%s' %(result[0],result[1])
            else:
                label = '%s%s->%s%s' %(result[0],result[1],result[2],result[3])

            if label not in data:
                data[label] = coop[:,2*i+offset]
            else:
                data[label]+= coop[:,2*i+offset]
            ndata[label] += 1

        # get mean %
        for label in data:
            if label == 'energy': continue
            data[label] = data[label] / ndata[label]

        return pd.DataFrame(data)

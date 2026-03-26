
__contributor__ = 'Kun Zhou'
#================================================================
# baisc method for control the input and output 
#================================================================

import os
import numpy as np

ClusterConfigs = ['mem','nprocshared','nproc','chk']


class GaussianIO(object):

    @classmethod
    def write_structure(cls, structure, path:str, **kwargs):

        with open(path, 'a+') as f:
            for atom in structure.atomic_positions:
                f.write("{0:6} {1[0]:>16.6}{1[1]:>16.6}{1[2]:>16.6}\n".format(atom.specie, atom.coord))

    @classmethod
    def write_input(cls, incar, path:str, **kwargs):
        from copy import deepcopy

        incar = deepcopy(incar)

        with open(path, 'w') as f:
            # cluster settings
            for key in ClusterConfigs:
                if key in incar:
                    f.write('%'+'%s=%s\n' %(key, incar.pop(key)))

            print(incar)
            # task & xcfunc
            if 'print_level' in kwargs:
                f.write('#%s %s %s\n' %(kwargs['print_level'], incar.pop('task'), incar.xc_func))
            else:
                f.write('# %s %s\n' %(incar.pop('task'), incar.xc_func))

            # null & label
            f.write('\n%s\n\n' %incar.label)

            # charge & structure
            f.write('%d %d\n' %(incar.charge, incar.mspin))
            for atom in incar.structure.atomic_positions:
                f.write("{0:2}{1[0]:>14.6f}{1[1]:>14.6f}{1[2]:>14.6f}\n".format(atom.specie, atom.coord))
            f.write("\n")
            '''
            if incar.structure.connections is not None:
                for atom1,value in incar.structure.get_connections(type='gaussian').items():
                    f.write("\n%d" %(atom1+1))
                    for atom2,num in value.items():
                        f.write(" %d %.1f" %(atom2+1,num))
            '''


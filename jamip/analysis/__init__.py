# coding: utf-8
# Copyright (c) JAMIP Development Team.
# Distributed under the terms of the JLU License.
#=================================================================
# This file is part of JAMIP.
#
# Copyright (C) 2017 Jilin University
#
#  JAMIP is a platform for high throughput calculation. It aims to
#  make simple to organize and run large numbers of tasks on the
#  superclusters and post-process the calculated results.
#
#  JAMIP is a useful packages integrated the interfaces for ab initio
#  programs, such as, VASP, Guassian, QE, Abinit and
#  comprehensive workflows for automatically calculating by using
#  simple parameters. Lots of methods to organize the structures
#  for high throughput calculation are provided, such as alloy,
#  heterostructures, etc.The large number of data are appended in
#  the MySQL databases for further analysis by using machine
#  learning.
#
#  JAMIP is free software. You can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published
#  by the Free sofware Foundation, either version 3 of the License,
#  or (at your option) and later version.
#
#  You should have recieved a copy of the GNU General Pulbic Lincense
#  along with JAMIP. If not, see <https://www.gnu.org/licenses/>.
#=================================================================
import os
import re
import numpy as np
from os.path import exists,join

class IterableAdapter:
    """https://stackoverflow.com/a/39564774"""
    def __init__(self, iterable_factory, length=None):
        self.iterable_factory = iterable_factory
        self.length = length

    def __iter__(self):
        return iter(self.iterable_factory())

class Finder:


    __task__ = None
    __builder__ = None
    __multi__ = False
    _stdin = None

    def __init__(self,stdin=None):
        self.stdin = stdin

    def grep(self,value):
        '''
        grep base information from scf calculation
        '''

        from jamip.analysis.vasp.outcar import GrepOutcar
        from jamip.analysis.qe.qexml import GrepXml

        if self.__builder__ == 'vasp':
            func = getattr(GrepOutcar(),value)
        elif self.__builder__ == 'qe':
            func = getattr(GrepXml(),value)
        elif self.__builder__ == 'jamip':
            func = getattr(GrepOutcar(),value)
        elif self.__builder__ == None:
            raise ValueError('Use the grep module before setting up the software.')
        return func(self.stdscf)

    @property
    def task(self):
        return self.__task__
        
    @property
    def stdin(self):
        return self._stdin

    @stdin.setter
    def stdin(self,path):
        path = self.seek(path)
        if path != None:
            self._stdin = path
        else:
            raise IOError("Missing calculation files!")

    @property
    def stdscf(self):

        if self.__builder__ == 'jamip':
            if exists(join(self._stdin, 'scf', 'OUTCAR')):
                return join(self._stdin, 'scf')
            elif exists(join(self._stdin, 'qerun', 'scf.xml')):
                return join(self._stdin, 'qerun', 'scf.xml')
        elif self.__multi__ == False:
            return self._stdin
        else:
            raise IOError('Seek scf directoty failed!')

    @property
    def stdcell(self):

        if self.__builder__ == 'jamip':
            contcar = join(self._stdin, 'scf', 'CONTCAR')
            poscar = join(self._stdin, 'scf', 'POSCAR')
        elif self.__multi__ == False:
            contcar = join(self._stdin, 'CONTCAR')
            poscar = join(self._stdin, 'POSCAR')
        else:
            raise IOError('Seek scf directoty failed!')

        if exists(contcar):
            return contcar
        elif exists(poscar):
            return poscar
        else:
            raise OSError('Seek std_structure failed!')

    def seek(self,path):
        '''
        get calculation type, return absolute path if exists
        '''
        from os.path import normcase, isdir, isfile

        path = normcase(path)
        if isdir(path):
            if exists(join(path,'.status')):
                self.__builder__ = 'jamip'
            elif exists(join(path,'OUTCAR')):
                self.__builder__ = 'vasp'
            else:
                path = None

        elif isfile(path) and path.endswith('.xml'):
            self.__builder__ = 'qe'

        else:
            path = None

        return path
















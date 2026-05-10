import os
import numpy as np
from dataclasses import dataclass
from jamip.abtools.vasp.check import CheckStatus

@dataclass
class Amset_Settings:
            
    distance = 0.005

class Amset:

    settings = Amset_Settings
    #requires = ['dielectric']#'deformation','elastic', 'dielectric']
    requires = ['deformation']#,'elastic', 'dielectric']
 
    def __init__(self, builder):
        self.obj = builder

    def __getattr__(self, attr):
        return getattr(self.obj, attr)

    def diy_calculator(self):

        task_id = 'amset'
        # set stdin & stdout %
        stdin = None
        if len(self.links[task_id]):
            stdin = self.tasks[self.links[task_id][0]].path

        # add parameters %
        incar = self.tasks[task_id]
        self.settings = incar.data
        self.clear_status(task_id)

        # deformaion %
        self.deformation_calculator(stdin)
        # elastic %
        #self.other_calculator('elastic')
        # dfpt %
        #self.other_calculator('dielectric')

        # check %
        if self.check(self.rootdir):
            status = {'task':'deform','finish':True,'success':True}
            self.write_status(status, self.rootdir/"electric"/"deform")
            status = {'task':'amset','finish':True,'success':True}
            self.write_status(status, self.rootdir/"electric"/"amset")
            incar.state = 'C'
        else:
            incar.state = 'E'

    def other_calculator(self, task_id):
        status = CheckStatus.load_status(self.rootdir)
        if task_id in status and status[task_id]['status']:
            return
             
        if len(self.links[task_id]) == 0:
            self.links[task_id].append(self.links['amset'][0])
        self.run(task_id)

    def deformation_calculator(self, stdin):
        from amset.deformation.generation import get_deformations, get_deformed_structures
        from pymatgen.core.tensors import symmetry_reduce
        from pymatgen.util.string import unicodeify_spacegroup
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from jamip.structure.convert import mp2jamip, jamip2mp

        # load status
        status = CheckStatus.load_status(self.rootdir)
        task_id = 'deform'
        if task_id in status and status[task_id]['status']:
            return

        # get bandedge & koints 
        stdout = self.rootdir/'electric'/'deform'
        distance = self.settings.get('distances', 0.005)
        symprec = self.settings.get('symprec', 1e-2)

        incar = self.tasks['deformation']
        incar.name = 'deform'
        incar.structure = self.load_structure(stdin)
        incar.kpoints = incar.kpoints.get_gamma_kpoints(
            cell=incar.structure.lattice,
            model=self.func.force_create_kpoints
            )

        structure = jamip2mp(incar.structure)

        deformations = get_deformations(distance)
        sga = SpacegroupAnalyzer(structure, symprec=symprec)            
        spg_symbol = unicodeify_spacegroup(sga.get_space_group_symbol())
        spg_number = sga.get_space_group_number()   
        deformations = list(symmetry_reduce(deformations, structure, symprec=symprec))
        deformed_structures = get_deformed_structures(structure, deformations)

        subtasks = []
        # cal base structure %
        output = stdout / 'scf'
        tmp = incar.get("lwave", "None")
        incar['lwave'] = True
        self.set_input(incar, output)
        subtasks.append(output)
        incar['lwave'] = tmp

        # cal deform structures %
        for i,cell in enumerate(deformed_structures):
            incar.structure = mp2jamip(cell)
            output = stdout / ('deforma-%d' %(i+1))
            self.set_input(incar, output)
            subtasks.append(output)

        # finally %
        self.batch_calculator('deformation', subtasks)

    @classmethod
    def check(cls, path):
        status = CheckStatus.load_status(path)

        result = True
        for task in ('deform', 'elastic', 'dielectric'):
            if task in status and status[task]['status']:
                pass
            else:
                result = False
                break

        return result

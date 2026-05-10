import os
from jamip.utils.phase import PhaseAnalysis
from jamip.structure import read, write, Structure
import numpy as np
import pandas as pd

class RestAPI:

    def __init__(self):
        pass

class Oqmd(RestAPI):

    def __init__(self, key:str=''):

        pass

    def get_phase_by_elements(self,elements:list,stable=True):
        import qmpy_rester as qr

        kwargs = {'composition': '-'.join(elements)}
        if stable == True:
            kwargs['stability'] = '0'

        with qr.QMPYRester() as q:
            list_of_data = q.get_oqmd_phases(verbose=False, **kwargs)

        composition = []
        energy = []
        spacegroup = []
        oqmd_id = []
        icsd_id = []
        name = []
        for data in list_of_data['data']:
            composition.append(data['composition'])
            energy.append(data['delta_e'])
            spacegroup.append(data['spacegroup']) 
            oqmd_id.append(data['formationenergy_id'])
            icsd_id.append(data['icsd_id'])
            name.append(data['name'])
             
        df = pd.DataFrame({'formula': composition,
                           'name': name,
                           'energy':energy,
                           'spacegroup':spacegroup,
                           'oqmd_id':oqmd_id,
                           'icsd_id':icsd_id,
                          })
        return df


    def get_structure_by_id(self, oqmd_id: int):
        import qmpy_rester as qr

        with qr.QMPYRester() as q:
            data = q.get_optimade_structure_by_id(oqmd_id)
            if data != None:
                structure = data['data']['attributes']
            if structure == None:
                raise RuntimeError('wget oqmd failed.')

            lattice = structure['lattice_vectors']
            elements = structure['species_at_sites']
            positions = structure['cartesian_site_positions']
            s = Structure.from_cell((lattice, positions, elements), direct=False)
            s.comment_line = 'Provided by oqmd %d' %oqmd_id
        return s


class Pymatgen(RestAPI):

    def __init__(self, key:str):

        self.key = 'BcMDH2aGWtT0Wpj4PcIV'
        pass


    def get_by_formula(self, formula:str, stable=True):
        from pymatgen.ext.matproj import MPRester

        with MPRester(self.key) as m:                                       
            data = m.get_data(formula)                                 
            energys = []                                               
            files = []
            mpids = []
            for value in data:
                ids = value['task_ids']
                print(ids)
                mpid = ids[0].split('-')[-1] 
                
                energys.append(value['formation_energy_per_atom'])
                files.append(value['cif'])
                mpids.append(mpid)


        

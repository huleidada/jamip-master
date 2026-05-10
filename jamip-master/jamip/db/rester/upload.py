from jamip.analysis.base import FinderSet
from ..materials.structure import Structure
from ..materials.composition import Composition
from ..materials.element import Element
from ..materials.species import Species
from ..materials.atom import Atom
from ..materials.entry import Entry
from ..materials.spacegroup import Spacegroup
from ..materials.prototype import Prototype
from ..iostream.read import Read
import pandas as pd
import pathlib
import os

class Upload(object):

    def __init__(self):
        self.__entry = 'vasp'

    def load_structure(self,path=None):
        '''
        support filetype: poscar, contcar, .xyz, .vasp, .mol, .cif
        '''        
        if path == None: path = pathlib.Path.cwd()
        finder = FinderSet(path=path, task='structure')

        # seek from files %
        for file in finder.stdin:
            try:   
                raw=Read(file).run()
            except:
                print("Read Structure %s into DB failed." %file)
                continue            
            Structure().create(raw, isPersist=True)
            print("Save Structure %s into DB." %file)

    def load_entry(self, path=None, properties=None, soft='vasp'):
        if path == None: path = pathlib.Path.cwd()
        finder = FinderSet(path=path, task='structure', soft=soft)
 
        sum = 0
        for path in finder.stdin: 
            EntryBuilder().load(path,properties,isPersist=True)
            sum += 1
            
        if sum == 0:
            print('No Valid calculation directory !')
        else:
            print('Total = %d' %sum)

class EntryBuilder(object):

    cal_params = ['date','datetime','vasp_version','prec','ediff','ediffg']

    def __init__(self,name='',path=None):
        self.entry = Entry()
        self.entry.name = name
        self.entry.path = path

    def load(self, root, properties=None, isPersist=False):
        # from jamip.abtools.vasp.check import CheckStatus
        from jamip.analysis.vasp import Finder

        root = pathlib.Path(root)

        if len(self.entry.name) == 0:
            self.entry.name = root.name
            self.entry.path = str(root.resolve())

        vasp = Finder(root)

        # load scf structure %
        raw = Read(vasp.stdcell, dtype='vasp').run()
        self.entry.structure = Structure().create(raw, isPersist)

        # load properties %
        if properties == None:
            properties = [field.name for field in self.entry._meta.fields]
        print(properties)

        # calculated_parameters
        if 'calculated_parameters' in properties:
            cal_params = {}
            for key in self.cal_params:
                try:
                    cal_params[key] = vasp.grep(key)
                except:
                    cal_params[key] = None
            self.entry.calculated_parameters = cal_params

        # free_energy
        if 'energy' in properties:
            self.set_energy(vasp.stdscf)

        if 'bandgap' in properties:
            self.set_bandgap(root)

        if 'boltztrap' in properties:
            self.set_boltztrap(root)

        if 'effective_mass_of_bandside' in properties:
            self.set_emass(root)

        if 'born_effective_charge' in properties:
            self.set_born_charge(root)

        self.entry.save()

    def set_born_charge(self,root):
        from jamip.analysis.vasp.outcar import GrepOutcar 
        path = root / 'electric' / 'born'
        if not path.exists(): return 0
        try:
            self.entry.born_effective_charge = {}
            for i,array in enumerate(GrepOutcar().born(path)):
                self.entry.born_effective_charge[i] = array
        except:
            print('load born_charge error')

    def set_boltztrap(self, root):
        from jamip.analysis.vasp.boltztrap import BoltztrapFinder
        try:
            H_mass,e_mass = BoltztrapFinder(root).get_effective_mass()
            self.entry.boltztrap = {'emass':e_mass, 'hmass':H_mass}
        except:
            print('load boltztrap error')

    def set_emass(self,root):
        from jamip.analysis.vasp import BandFinder
        try:
            emass = BandFinder(root).get_emass()
            emass_dict = {}
            cbm = [mass for key,mass in emass.items() if 'cbm' in key]
            vbm = [mass for key,mass in emass.items() if 'vbm' in key]
            if len(cbm):
                emass_dict['hmass'] = round(len(cbm)/sum([1/i for i in cbm]),4)
            if len(vbm):
                emass_dict['emass'] = round(len(vbm)/sum([1/i for i in vbm]),4)
            self.entry.effective_mass_of_bandside = emass_dict 
        except:
            print('load emass error')
        
    def set_bandgap(self,root):
        from jamip.analysis.vasp import BandFinder
        try:
            self.entry.bandgap = BandFinder(root).get_bandgap()
        except:
            print('load bandgap error')

    def set_energy(self,path):
        from jamip.analysis.vasp.outcar import GrepOutcar
        try:
            free_energy = GrepOutcar().free_energy(path)
            self.entry.energy = free_energy
            if not (self.entry.structure is None):
                self.entry.energy_per_formula = free_energy/self.entry.structure.multiple
                self.entry.energy_per_atom = free_energy/self.entry.structure.natoms
        except:
            print('load energy error')
            


class SqliteSQL:

    __db__ = 'sqlite3'

    def __init__(self):
        pass

    def sql_by_id(self,table,mainkey=None):
        dataset = pd.DataFrame(columns=['name','format','natoms','SG'])
        if table == 'entry':
            fields = [f.attname for f in Entry._meta.fields]
            if mainkey == None:
                query = Entry.objects.all()
            else:
                #query = Entry.objects.filter(name=mainkey.encode('utf-8'))
                query = Entry.objects.filter(name=mainkey)
                
            if len(query) == 0:
                mainkey = 'all' if mainkey is None else mainkey
                print('Query failed for %s in table %s. Exit!' %(mainkey,table))
                return
            dataset.loc[0] = [query[0].name,
                              query[0].structure.composition.formula,
                              query[0].structure.natoms,
                              query[0].structure.spacegroup]
            print(dataset)
        while True:
            line = input("([fileds]/all/none):")
            if line == 'all':
                print(fields)
            elif len(line):
                for key in line.split():
                    try:
                        result = getattr(query[0],key)
                        if isinstance(result,dict):
                            for k,v in result.items():
                                dataset[k] = v
                        else:
                            dataset[key] = result 
                    except:
                        pass
                print(dataset)
            else:
                return 

    def load_history(self):
        return self.sql_by_id(table='entry')

class __DBShell(Upload,SqliteSQL):

    def __init__(self, params=None, *args, **kwargs):
        super().__init__()
        path = None
        if 'pool' in params:
            path = os.path.abspath(params['pool']) 
        
        if params['db']  == 'structure':
            self.load_structure(path)
        elif params['db'] == 'entry':
            self.load_entry(path)
        elif params['db'] == 'history':
            self.load_history()


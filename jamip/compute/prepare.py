# prepare
import os
import logging
from .pool import Pool
from jamip.utils.logger import load_yaml, dump_yaml, full_path 

class Prepare(object):
    """
    The Prepare class provides a set of class methods to facilitate the preparation of computational workflows, including resource pooling, cluster configuration, INCAR file management, task linking, and database status checking.
    Class Methods
    -------------
    - pool(func):
        Wraps a function with PoolTools for parallel execution.
    - cluster(system='pbs'):
        Initializes and manages cluster configuration files using templates and YAML files. Ensures the correct cluster manager is set, copies templates if needed, and checks for undefined parameters in Jinja2 templates.
    - incar(func, **kwargs):
        Prepares and updates INCAR configuration files for computational tasks. Merges default, local, and inherited parameters, and writes the final configuration to '.incar'.
    - links(func, **kwargs):
        Generates and manages task dependency links, writing them to '.link'. Ensures necessary dependencies are present and warns about missing tasks.
    - checkdb(db='mysqld'):
        Checks if the specified database process is running for the current user and prints a warning if it is not.
    """
       
    @classmethod
    def pool(cls,func):
        return PoolTools(func)

    @classmethod
    def cluster(cls,system='pbs'):
        import shutil
        import jinja2
        from jinja2.meta import find_undeclared_variables
        
        system = system.lower()
        params = load_yaml('.cluster')

        # check %
        if params != None: 
            if params["manager"].lower() != system:
                os.rename('.cluster', f'.{params["manager"].lower()}')
                logging.info(f"Cluster changed from {params['manager'].lower()} to {system} !")

        if not os.path.exists('.cluster') or params is None: 
            template = full_path(f'~/.jamip/env/{system}.yaml') 
            shutil.copy(template,'.cluster')
            logging.info(f"Initialize cluster with {system}.yaml.")
            params = load_yaml('.cluster')

        # check template %
        envdir = full_path("~/.jamip/env")
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(envdir), undefined=jinja2.DebugUndefined)
        rendered = env.get_template(f'{system}.template').render(params)
        ast = env.parse(rendered)
        undefined = find_undeclared_variables(ast) 
        for key in undefined:
            if key not in ('outdir', 'output', 'script', 'pool', 'root'):
                logging.error(f"The template file contains undefined parameters: {key}")
        # if undefined: raise jinja2.UndefinedError(f'The following variables are undefined: {undefined!r}')

    @classmethod
    def incar(cls, func, **kwargs):
        from copy import deepcopy

        template = full_path(f'~/.jamip/env/{func.soft}.yaml')
        default_incar = load_yaml(template)
        local_incar = load_yaml('.incar')
        if local_incar is None: local_incar = {}
        future_incar = deepcopy(func.tasks)
        all_tasks = func.get_all_tasks()

        # base 
        if 'base' in local_incar:
            future_incar['base'] = local_incar['base']
        elif 'base' in default_incar:
            future_incar['base'] = default_incar['base']
        future_incar.move_to_end('base', last=False)

        # full tasks %
        for task_id in all_tasks:
            if task_id not in future_incar:
                future_incar[task_id] = {} 

            if task_id in local_incar:
                future_incar[task_id].update(local_incar[task_id])
            elif task_id in default_incar:
                future_incar[task_id].update(default_incar[task_id])
            elif len(future_incar[task_id]) == 0:
                logging.warning(f'Missing {task_id} params in template.')

        # inherits params 
        for task_id in local_incar:
            if task_id not in future_incar and len(local_incar[task_id]):
                future_incar[task_id] = local_incar[task_id]
                
        # build incar.yaml %
        dump_yaml(dict(future_incar), '.incar')

    @classmethod
    def links(cls, func, **kwargs):

        links_dict = {}

        for task_id in func.tasks:
            if task_id in func.links:
                links_dict[task_id] = func.links[task_id]
            else:
                links_dict[task_id] = []

        # scf reset %
        if 'scf' in links_dict and len(links_dict['scf']) == 0:
            if 'relax' in links_dict and len(links_dict['relax']) == 0:
                links_dict['scf'] = ['relax']

        # build link.yaml %
        dump_yaml(links_dict, '.link')

        # warn %
        rev = []
        for root,children in links_dict.items():
            rev.append(root)
            rev.extend(children)
        for task_id in set(rev):
            if task_id not in func.tasks:
                logging.warning('Missing necessary dependency calculations: %s' %task_id)

         
    @classmethod
    def checkdb(cls,db='mysqld'):
        user = os.environ['USER']
        dbrun = True
        # database check %
        lines = os.popen("ps -ef|grep %s" %db).readlines()
        for line in lines:
            if line.startswith(user) and line.split()[2] == '1':
                dbrun = False
        if dbrun:
            print(f"Database {db} is not running")

class PoolTools:

    def __init__(self, func):
        self.pool = Pool()
        self.pool.functional = func

    def set_structure(self, input, outdir='./Results'):
        from copy import deepcopy
        from jamip.structure import read
        import six, sys

        # check
        input = full_path(input)
        # tasks class to dict %

        for i in os.walk(input).__next__()[2]:
            func = deepcopy(self.pool.functional)
            try:
                func.structure = read(os.path.join(input,i))
            except Exception as err:
                print(os.path.join(input,i))
                value = sys.exc_info()
                six.reraise(*value)
            self.pool.add_tasks(os.path.join(outdir,i), func)


    def set_extra(self, *args):
        '''
        create magmom configuration file with default format
        '''
        if len(args) == 0:
            args = ('magmom',) 

        if 'ldau' in args:
            args += ('ldaul','ldauu','ldauj')

        params = load_yaml('.extra') 
        if params == None: params = {}

        with open('.extra','w') as f:
            if 'magmom' in args: 
                spin_type = 'non-colinear' if 'soc' in self.pool.functional.xc_func else 'colinear'
                f.write('# %s magnetic moment\n' %spin_type)

            for key,fmt in self.pool.mainkey.items():
                if key == 'ldau': continue
                f.write(f'{key} : # {fmt}\n')
                for param in args:
                    f.write(f'   {param} : ')
                    if key in params and params[key].get(param, None) != None:
                        f.write(str(params[key][param]))
                    f.write('\n')
            
    def set_potential(self):
        '''
        create potential configuration file with default format
        '''
        params = load_yaml('.potential') 
        if params == None: params = {}

        maps = {}
        for key,value in self.pool._pool.items():
            maps[key] = {'species': list(value.structure.species_of_elements),}
            if key in params:
                maps[key]['files'] = params[key]
            else:
                maps[key]['files'] = value.get_potential_files(value.structure)

        with open('.potential','w') as f:
            for key,value in maps.items():
                f.write(f'{key}: # {" ".join(value["species"])}\n')
                for file in value['files']:
                    f.write(f'-  {file}\n')

    def save(self,file ='pool/jamip', mode='n', prior=9):

        basestatus = {'status': 'W', 'prior': prior, 'id': '%8s'%-1}  
        self.pool.save(file, mode)    
        with self.pool.open(file, mode) as pool:
            for key in self.pool.joblist:
                pool[key] = basestatus
                print(pool.values())
 

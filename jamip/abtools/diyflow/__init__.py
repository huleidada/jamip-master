import importlib

def get_diy_modules():
    '''
    Get diy_module in the current directory.
    '''
    import importlib.resources
    modules = []
    for module in importlib.resources.contents(__name__):
        if module.endswith('.py'):
            modules.append(module.split('.')[0])
    return modules

def import_diy_module(module_name):
    '''
    Calls the corresponding module from the current 
    directory based on the module name
    '''
    if module_name not in get_diy_modules():
        raise ImportError("Moudle %s not exists" %module_name)

    diy_module = importlib.import_module('jamip.abtools.diyflow.'+module_name)
    diy_class = getattr(diy_module,module_name.capitalize())

    return diy_class

def get_requires(tasks):

    diy_tasks = get_diy_modules()
    base_requires = {'hse_gap': 'hse', 'hse_band': 'hse',
                    'nve': 'md', 'nvt': 'md', 'npt': 'md',
                    'mlmd': 'md', 'grunesien': 'fc2',}

    requires = []
    for task in tasks:
        if task in base_requires:
            requires.append(base_requires[task])
        if task in diy_tasks:
            diy_class = import_diy_module(task)
            if hasattr(diy_class, 'requires'):
                requires.extend(diy_class.requires)
        
    return list(set(requires))

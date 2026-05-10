from jamip.utils.logger import run_logger

def get_package_path():
    import jamip
    import os.path
    return os.path.dirname(jamip.__file__)

def main():
    from .parse2 import __ParseProcess

    jamip = __ParseProcess()
    parse = jamip.__collect_parses__()
   
    if len(parse) ==0:
        print('jp -h/--help')
        print(get_package_path())
        return

    print(parse)
 
    if 'run' in parse:
        from jamip.compute import LaunchTasks
        run_logger()
        LaunchTasks(parse) 

    elif 'vasp_tools' in parse:
        from .vasptools import __Vasptools
        __Vasptools().run(parse)	

    elif 'cp2k_tools' in parse:
        from .cp2ktools import __CP2ktools
        __CP2ktools().run(parse)	

    elif 'output' in parse:
        from .output import __OutputData
        __OutputData().run(parse)

    elif 'plot' in parse:
        from jamip.utils.plot import Plot
        pl = Plot(path='./', soft=parse['soft'])
        pl.plots(*parse['plot'])

    elif 'db' in parse:
        from jamip.db.connect import __DBShell
        __DBShell(parse)
	
    elif 'check' in parse:
        from .check import __CheckStatus
        __CheckStatus(parse)

    elif 'input' in parse:
        from .create import __CreateInput
        __CreateInput(parse)

    elif 'mysql' in parse:
        from .softmanage import __Mysql
        __Mysql(parse)

    if 'django' in parse:
        from .softmanage import __Django
        __Django(parse)

    if 'phonon' in parse:
        from .phonon import PhononTools
        PhononTools(parse) 

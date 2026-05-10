import os
import numpy as np
from pathlib import Path
import shutil
from jamip.structure import read,write,Structure

class Tdep:

    def __init__(self, builder):
        self.obj = builder

    def __getattr__(self, attr):
        return getattr(self.obj, attr)

    def diy_calculator(self):
        from jamip.analysis.vasp.md import TDEP 

        task_id = 'tdep'
        # set stdin & stdout %
        stdin = None
        if len(self.links[task_id]):
            stdin = self.tasks[self.links[task_id][0]].path
        stdout = self.rootdir / 'md' / 'tdep'
        incar = self.tasks[task_id]
        self.clear_status(task_id)

        # write TDEP inputs parameters %
        skip_steps = incar.get("skip_steps", 3000)
        infile = Path(stdin)/"OUTCAR"
        model = TDEP(stdout, skip_steps)
        model.load_info(infile)
        model.load_timestep(infile)
        model.write_meta(workdir=stdout)

        # write TDEP structures %
        if (Path(stdin)/"infile.ssposcar").exists():
            shutil.copy(Path(stdin)/"infile.ssposcar", stdout/"infile.ssposcar")
            shutil.copy(Path(stdin)/"infile.ucposcar", stdout/"infile.ucposcar")
        else:
            shutil.copy(Path(stdin)/"POSCAR", stdout/"infile.ssposcar")
            shutil.copy(Path(stdin)/"POSCAR", stdout/"infile.ucposcar")

        # run extract_forceconstants %
        command = "extract_forceconstants"
        for key,value in incar.items():
            if key.lower() == "rc2":
                command += f" -rc2 {value}"
            elif key.lower() == "rc3":
                command += f" -rc3 {value}"
            elif key.lower() == "rc4":
                command += f" -rc4 {value}"
            elif key.lower() == "s":
                command += f" -s {value}"
        print(command)

        os.chdir(stdout)
        #os.popen(f"{command} > ef.log").readline()
        os.chdir(self.rootdir)

        # check forceconstant %
        if not (stdout/'outfile.forceconstant').exists():
            incar.state = 'E'
            return

        # run phonon_dispersion_relations %
        os.chdir(stdout)
        os.popen(f"phonon_dispersion_relations > pdr.log").readline()
        print("phonon_dispersion_relations")
        os.chdir(self.rootdir)

        # run thermal_conductivity %
        for suffix in [".forceconstant",".forceconstant_thirdorder",".forceconstant_fourthorder"]:
            if (stdout/f"outfile{suffix}").exists(): 
                shutil.copy(stdout/f"outfile{suffix}", stdout/f"infile{suffix}")
        temp = incar.get("temperature", 300)
        command = f"thermal_conductivity -qg 10 10 10 --temperature {temp}"
        print(command)

        os.chdir(stdout)
        os.popen(f"{command} > tc.log").readline()
        os.chdir(self.rootdir)

        raise
        status = {'task':'tdep','finish':True,'success':True}
        self.write_status(status, stdout)
        incar.state = 'C'

    @classmethod
    def check(self,path):
        path = Path(path)
        if path.name == 'tdep':
            file = path/'outfile.forceconstant'
        else:
            file = path/'md'/'tdep'/'outfile.forceconstant'
        if file.exists() and file.st_size > 0:
            print('TDEP calculation finish.')
            return True
        else:
            return False

